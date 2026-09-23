from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from tests.conftest import create_expedition, create_route, create_user
from trailforge.config import Settings
from trailforge.database.migrations import initialize_database
from trailforge.database.session import Database
from trailforge.domain.enums import InventoryMovementType
from trailforge.errors import IdempotencyConflictError, InventoryError
from trailforge.main import create_app
from trailforge.models.activities import ExpeditionRegistration
from trailforge.models.audit import AuditLog, IdempotencyRecord
from trailforge.models.gear import GearInventory, GearLoan, InventoryMovement
from trailforge.models.safety import ItineraryCheckIn
from trailforge.schemas.gear import (
    GearCatalogCreate,
    GearInventoryCreate,
    GearLoanCreate,
    InventoryAdjustment,
)
from trailforge.schemas.safety import CheckInScheduleCreate, CheckInSubmit
from trailforge.services.gear import GearService
from trailforge.services.safety import SafetyService

BEIJING = timezone(timedelta(hours=8))


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


# --------------------------------------------------------------------------- helpers


def _make_inventory(database: Database, *, quantity: int = 5) -> tuple[int, int, int]:
    with database.session() as session:
        owner = create_user(session, email="owner@example.com", name="Owner")
        service = GearService(session)
        catalog = service.create_catalog(
            GearCatalogCreate(
                sku="ROPE-60M",
                name="60m climbing rope",
                category="safety",
                default_weight_grams=3800,
                safety_critical=True,
            ),
            actor_id=owner,
        )
        inventory = service.create_inventory(
            GearInventoryCreate(
                catalog_id=catalog.id,
                ownership="club",
                quantity_total=quantity,
                condition="good",
                actor_id=owner,
                idempotency_key="seed-inventory-key",
            )
        )
        inventory_id = inventory.id
    return owner, catalog.id, inventory_id


def _make_check_in(database: Database) -> tuple[int, int, int, datetime]:
    """Returns organizer_id, expedition_id, check_in_id, due_at."""
    with database.session() as session:
        organizer = create_user(session, email="leader@example.com", name="Leader")
        route = create_route(session, actor_id=organizer)
        expedition_id = create_expedition(session, organizer_id=organizer, route_id=route)
        due = GearService(session).expeditions.get(expedition_id).start_at + timedelta(hours=1)
        check_in = SafetyService(session).schedule_check_in(
            expedition_id,
            CheckInScheduleCreate(user_id=organizer, check_in_type="routine", due_at=due),
            actor_id=organizer,
        )
        return organizer, expedition_id, check_in.id, due


def _row_counts(database: Database, inventory_id: int) -> dict[str, int]:
    with database.session() as session:
        return {
            "movements": session.scalar(
                select(func.count())
                .select_from(InventoryMovement)
                .where(InventoryMovement.inventory_id == inventory_id)
            ),
            "loan_movements": session.scalar(
                select(func.count())
                .select_from(InventoryMovement)
                .where(
                    InventoryMovement.inventory_id == inventory_id,
                    InventoryMovement.movement_type == InventoryMovementType.LOAN_OUT,
                )
            ),
            "loans": session.scalar(
                select(func.count())
                .select_from(GearLoan)
                .where(GearLoan.inventory_id == inventory_id)
            ),
            "idem": _business_idem_count(session),
            "pending_idem": session.scalar(
                select(func.count())
                .select_from(IdempotencyRecord)
                .where(IdempotencyRecord.status == "pending")
            ),
        }


def _business_idem_count(session) -> int:
    """Idempotency rows excluding the one written by the inventory seed helper."""
    return session.scalar(
        select(func.count())
        .select_from(IdempotencyRecord)
        .where(IdempotencyRecord.scope != "gear:inventory:create")
    )


# ---------------------------------------------------- 1. concurrent same-key same-payload


@pytest.mark.parametrize("workers", [8])
def test_concurrent_inventory_adjustment_executes_once(database: Database, workers: int) -> None:
    owner, _, inventory_id = _make_inventory(database, quantity=3)
    barrier = threading.Barrier(workers)

    def call() -> tuple[int, dict] | Exception:
        with database.session() as session:
            barrier.wait()
            try:
                response = GearService(session).adjust_inventory(
                    inventory_id,
                    InventoryAdjustment(
                        quantity_delta=2,
                        reason="Recount after expedition",
                        actor_id=owner,
                        idempotency_key="concurrent-adjust-key",
                    ),
                )
                return response.version, response.model_dump(mode="json")
            except Exception as exc:  # captured and re-raised by the main thread
                return exc

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda _: call(), range(workers)))

    assert not any(isinstance(item, Exception) for item in results), [
        item for item in results if isinstance(item, Exception)
    ]
    bodies = [body for _, body in results]
    # Every caller received the *first* committed response, byte-identical.
    assert all(body == bodies[0] for body in bodies)
    assert bodies[0]["quantity_total"] == 5
    assert bodies[0]["quantity_available"] == 5

    with database.session() as session:
        inventory = session.get(GearInventory, inventory_id)
        assert inventory.quantity_total == 5
        assert inventory.quantity_available == 5
        movements = list(
            session.scalars(
                select(InventoryMovement)
                .where(InventoryMovement.inventory_id == inventory_id)
                .order_by(InventoryMovement.id)
            )
        )
        assert [m.movement_type for m in movements] == ["initial", "adjustment"]
        assert movements[-1].quantity_delta == 2
        assert movements[-1].quantity_after == 5
        assert (
            session.scalar(
                select(func.count())
                .select_from(AuditLog)
                .where(
                    AuditLog.entity_type == "gear_inventory",
                    AuditLog.correlation_id == "concurrent-adjust-key",
                )
            )
            == 1
        )
        records = list(
            session.scalars(
                select(IdempotencyRecord).where(
                    IdempotencyRecord.idempotency_key == "concurrent-adjust-key"
                )
            )
        )
        assert len(records) == 1
        assert records[0].status == "completed"
        assert records[0].resource_id == inventory_id
        assert records[0].response_json["quantity_total"] == 5


def test_concurrent_loan_executes_once_and_replay_freezes_snapshot(database: Database) -> None:
    owner, _, inventory_id = _make_inventory(database, quantity=4)
    with database.session() as session:
        borrower = create_user(session, email="borrow@example.com", name="Borrower")
    workers = 6
    barrier = threading.Barrier(workers)
    loaned_at = datetime.now(BEIJING)

    def call() -> dict | Exception:
        with database.session() as session:
            barrier.wait()
            try:
                return (
                    GearService(session)
                    .loan(
                        GearLoanCreate(
                            inventory_id=inventory_id,
                            borrower_id=borrower,
                            quantity=3,
                            loaned_at=loaned_at,
                            due_at=loaned_at + timedelta(days=3),
                            actor_id=owner,
                            idempotency_key="concurrent-loan-key",
                        )
                    )
                    .model_dump(mode="json")
                )
            except Exception as exc:
                return exc

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda _: call(), range(workers)))
    assert not any(isinstance(item, Exception) for item in results), results
    assert all(body == results[0] for body in results)
    first = results[0]
    assert first["status"] == "active"
    assert first["condition_out"] == "good"
    assert first["returned_quantity"] == 0
    # Timezone-normalized snapshot renders the +08:00 input as UTC.
    assert first["loaned_at"].rstrip("Z").endswith("+00:00") or first["loaned_at"].endswith("Z")
    assert first["loaned_at"].startswith(loaned_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S"))

    with database.session() as session:
        inventory = session.get(GearInventory, inventory_id)
        assert inventory.quantity_available == 1
        loan = session.scalar(select(GearLoan))
        assert loan.quantity == 3
        assert (
            session.scalar(
                select(func.count())
                .select_from(InventoryMovement)
                .where(
                    InventoryMovement.inventory_id == inventory_id,
                    InventoryMovement.movement_type == InventoryMovementType.LOAN_OUT,
                )
            )
            == 1
        )
        assert (
            session.scalar(
                select(func.count())
                .select_from(AuditLog)
                .where(AuditLog.correlation_id == "concurrent-loan-key")
            )
            == 1
        )

    # A later partial return mutates the loan; a replay must still return the
    # original first-response snapshot, not the resource's current state.
    with database.session() as session:
        loan_id = session.scalar(select(GearLoan.id))
        GearService(session).return_loan(
            loan_id,
            _return_payload(owner, loaned_at + timedelta(days=1), quantity=1, key="return-1"),
        )
    with database.session() as session:
        replay = GearService(session).loan(
            GearLoanCreate(
                inventory_id=inventory_id,
                borrower_id=borrower,
                quantity=3,
                loaned_at=loaned_at,
                due_at=loaned_at + timedelta(days=3),
                actor_id=owner,
                idempotency_key="concurrent-loan-key",
            )
        )
        assert replay.returned_quantity == 0
        assert replay.status == "active"
        assert replay.version == first["version"]
        current = session.get(GearLoan, loan_id)
        assert current.returned_quantity == 1
        assert current.version > replay.version

    counts = _row_counts(database, inventory_id)
    assert counts["loans"] == 1
    assert counts["loan_movements"] == 1
    # loan + its partial return each have a completed reservation; the loan key
    # replays did not create extras.
    assert counts["idem"] == 2
    assert counts["pending_idem"] == 0

def test_concurrent_check_in_submission_executes_once(database: Database) -> None:
    organizer, _, check_in_id, due = _make_check_in(database)
    workers = 6
    barrier = threading.Barrier(workers)
    checked_in_at = (due + timedelta(minutes=45)).astimezone(BEIJING)

    def call() -> dict | Exception:
        with database.session() as session:
            barrier.wait()
            try:
                return (
                    SafetyService(session)
                    .submit_check_in(
                        check_in_id,
                        CheckInSubmit(
                            checked_in_at=checked_in_at,
                            latitude=46.1234,
                            longitude=6.9876,
                            note="Summit reached",
                            is_safe=True,
                            idempotency_key="concurrent-checkin-key",
                        ),
                    )
                    .model_dump(mode="json")
                )
            except Exception as exc:
                return exc

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda _: call(), range(workers)))
    assert not any(isinstance(item, Exception) for item in results), results
    assert all(body == results[0] for body in results)
    first = results[0]
    assert first["late_minutes"] == 45
    assert _parse(first["checked_in_at"]) == checked_in_at.astimezone(UTC)
    assert first["is_safe"] is True
    assert first["check_in_type"] == "routine"

    with database.session() as session:
        check_in = session.get(ItineraryCheckIn, check_in_id)
        assert check_in.checked_in_at == checked_in_at.astimezone(UTC)
        assert check_in.late_minutes == 45
        assert (
            session.scalar(
                select(func.count())
                .select_from(AuditLog)
                .where(
                    AuditLog.entity_type == "itinerary_check_in",
                    AuditLog.correlation_id == "concurrent-checkin-key",
                )
            )
            == 1
        )
        records = list(
            session.scalars(
                select(IdempotencyRecord).where(
                    IdempotencyRecord.idempotency_key == "concurrent-checkin-key"
                )
            )
        )
        assert len(records) == 1
        assert records[0].status == "completed"
        assert records[0].response_json["late_minutes"] == 45

    # Repeated replay after commit still returns the frozen snapshot and writes
    # no further audit rows.
    for _ in range(3):
        with database.session() as session:
            again = SafetyService(session).submit_check_in(
                check_in_id,
                CheckInSubmit(
                    checked_in_at=checked_in_at,
                    latitude=46.1234,
                    longitude=6.9876,
                    note="Summit reached",
                    is_safe=True,
                    idempotency_key="concurrent-checkin-key",
                ),
            )
            assert again.late_minutes == 45
            assert again.checked_in_at == checked_in_at.astimezone(UTC)
    with database.session() as session:
        assert _business_idem_count(session) == 1
        assert (
            session.scalar(
                select(func.count())
                .select_from(AuditLog)
                .where(
                    AuditLog.entity_type == "itinerary_check_in",
                    AuditLog.correlation_id == "concurrent-checkin-key",
                )
            )
            == 1
        )


# ------------------------------------------------------- 2. same key, different payload


def test_same_key_different_payload_conflicts_stably(database: Database) -> None:
    owner, _, inventory_id = _make_inventory(database, quantity=10)
    key = "reused-different-payload"
    with database.session() as session:
        first = GearService(session).adjust_inventory(
            inventory_id,
            InventoryAdjustment(
                quantity_delta=1, reason="First", actor_id=owner, idempotency_key=key
            ),
        )
        assert first.quantity_total == 11

    def attempt(delta: int, reason: str) -> object:
        with database.session() as session:
            try:
                return GearService(session).adjust_inventory(
                    inventory_id,
                    InventoryAdjustment(
                        quantity_delta=delta,
                        reason=reason,
                        actor_id=owner,
                        idempotency_key=key,
                    ),
                )
            except Exception as exc:
                return exc

    for _ in range(4):
        result = attempt(2, "Different delta")
        assert isinstance(result, IdempotencyConflictError)

    # Original payload still replays successfully and keeps the first snapshot.
    replay = attempt(1, "First")
    assert not isinstance(replay, Exception)
    assert replay.quantity_total == 11

    with database.session() as session:
        assert _business_idem_count(session) == 1
        assert session.get(GearInventory, inventory_id).quantity_total == 11
        # No extra adjustment movement or audit from failed attempts.
        assert (
            session.scalar(
                select(func.count())
                .select_from(InventoryMovement)
                .where(
                    InventoryMovement.inventory_id == inventory_id,
                    InventoryMovement.movement_type == InventoryMovementType.ADJUSTMENT,
                )
            )
            == 1
        )


def test_concurrent_same_key_mixed_payload_all_conflict_or_replay(database: Database) -> None:
    owner, _, inventory_id = _make_inventory(database, quantity=10)
    key = "mixed-payload-race-key"

    def call(delta: int) -> object:
        with database.session() as session:
            try:
                return GearService(session).adjust_inventory(
                    inventory_id,
                    InventoryAdjustment(
                        quantity_delta=delta,
                        reason=f"delta-{delta}",
                        actor_id=owner,
                        idempotency_key=key,
                    ),
                )
            except Exception as exc:
                return exc

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(call, [3, 4, 4, 3]))
    successes = [r for r in results if not isinstance(r, Exception)]
    conflicts = [r for r in results if isinstance(r, IdempotencyConflictError)]
    assert successes, "exactly one payload must win the reservation"
    assert len(successes) + len(conflicts) == 4
    final_total = successes[0].quantity_total
    assert final_total in {13, 14}
    assert all(r.quantity_total == final_total for r in successes)
    with database.session() as session:
        assert session.get(GearInventory, inventory_id).quantity_total == final_total
        assert _business_idem_count(session) == 1
        # Side effects ran for the winner only.
        assert (
            session.scalar(
                select(func.count())
                .select_from(InventoryMovement)
                .where(
                    InventoryMovement.inventory_id == inventory_id,
                    InventoryMovement.movement_type == InventoryMovementType.ADJUSTMENT,
                )
            )
            == 1
        )


# ------------------------------------------------------------ 3. rollback recovery


def test_rolled_back_transaction_frees_the_key(database: Database) -> None:
    owner, _, inventory_id = _make_inventory(database, quantity=5)
    key = "rollback-recovery-key"

    # First attempt fails business validation inside the request transaction;
    # both the side effects and the pending placeholder must roll back.
    with pytest.raises(InventoryError, match="negative"), database.session() as session:
        GearService(session).adjust_inventory(
            inventory_id,
            InventoryAdjustment(
                quantity_delta=-9,
                reason="Impossible",
                actor_id=owner,
                idempotency_key=key,
            ),
            )

    with database.session() as session:
        assert _business_idem_count(session) == 0
        assert (
            session.scalar(
                select(func.count())
                .select_from(InventoryMovement)
                .where(
                    InventoryMovement.inventory_id == inventory_id,
                    InventoryMovement.movement_type == InventoryMovementType.ADJUSTMENT,
                )
            )
            == 0
        )

    # The key is immediately reusable for a valid request.
    with database.session() as session:
        response = GearService(session).adjust_inventory(
            inventory_id,
            InventoryAdjustment(
                quantity_delta=2,
                reason="Valid retry",
                actor_id=owner,
                idempotency_key=key,
            ),
        )
        assert response.quantity_total == 7

    with database.session() as session:
        records = list(
            session.scalars(
                select(IdempotencyRecord).where(IdempotencyRecord.idempotency_key == key)
            )
        )
        assert len(records) == 1
        assert records[0].status == "completed"
        assert records[0].response_json["quantity_total"] == 7
        assert session.get(GearInventory, inventory_id).quantity_total == 7


def test_orphaned_pending_placeholder_is_reclaimed(database: Database) -> None:
    """A process that died after committing a placeholder (crash scenario) must
    not lock the key forever: the next same-key request reuses the row."""
    owner, _, inventory_id = _make_inventory(database, quantity=5)
    key = "crashed-pending-key"
    with database.session() as session:
        # Simulate a process that reserved the placeholder in its own committed
        # write and died before running/completing the business transaction.
        lease = GearService(session).begin_idempotent(
            scope=f"gear:inventory:{inventory_id}:adjust",
            key=key,
            payload=InventoryAdjustment(
                quantity_delta=1, reason="Orphan", actor_id=owner, idempotency_key=key
            ),
        )
        assert lease.record_id is not None
    with database.session() as session:
        row = session.scalar(
            select(IdempotencyRecord).where(IdempotencyRecord.idempotency_key == key)
        )
        assert row is not None
        assert row.status == "pending"

    with database.session() as session:
        response = GearService(session).adjust_inventory(
            inventory_id,
            InventoryAdjustment(
                quantity_delta=1,
                reason="Orphan",
                actor_id=owner,
                idempotency_key=key,
            ),
        )
        assert response.quantity_total == 6
    with database.session() as session:
        records = list(
            session.scalars(
                select(IdempotencyRecord).where(IdempotencyRecord.idempotency_key == key)
            )
        )
        assert len(records) == 1
        assert records[0].status == "completed"
        assert records[0].response_json["quantity_total"] == 6


# ------------------------------------------------------------ 4. application restart


def test_replay_survives_application_restart(tmp_path) -> None:
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'restart.db'}",
        sqlite_timeout_seconds=5,
        sqlite_busy_retries=2,
        sqlite_busy_backoff_seconds=0.01,
    )
    first_db = Database(settings)
    initialize_database(first_db)
    try:
        owner, _, inventory_id = _make_inventory(first_db, quantity=6)
        organizer, expedition_id, check_in_id, due = _make_check_in(first_db)
        with first_db.session() as session:
            borrower = create_user(session, email="restart-borrow@example.com", name="Reboot")
        loaned_at = datetime.now(BEIJING)
        with first_db.session() as session:
            loan = GearService(session).loan(
                GearLoanCreate(
                    inventory_id=inventory_id,
                    borrower_id=borrower,
                    quantity=2,
                    loaned_at=loaned_at,
                    due_at=loaned_at + timedelta(days=2),
                    actor_id=owner,
                    idempotency_key="restart-loan-key",
                )
            )
            loan_body = loan.model_dump(mode="json")
        with first_db.session() as session:
            checkin = SafetyService(session).submit_check_in(
                check_in_id,
                CheckInSubmit(
                    checked_in_at=due + timedelta(minutes=12),
                    note="Before reboot",
                    is_safe=True,
                    idempotency_key="restart-checkin-key",
                ),
            )
            checkin_body = checkin.model_dump(mode="json")
    finally:
        first_db.engine.dispose()

    # Simulate an application restart: brand new engine against the same file.
    second_db = Database(settings)
    initialize_database(second_db)
    try:
        with second_db.session() as session:
            replayed_loan = GearService(session).loan(
                GearLoanCreate(
                    inventory_id=inventory_id,
                    borrower_id=borrower,
                    quantity=2,
                    loaned_at=loaned_at,
                    due_at=loaned_at + timedelta(days=2),
                    actor_id=owner,
                    idempotency_key="restart-loan-key",
                )
            )
            assert replayed_loan.model_dump(mode="json") == loan_body

            replayed_checkin = SafetyService(session).submit_check_in(
                check_in_id,
                CheckInSubmit(
                    checked_in_at=due + timedelta(minutes=12),
                    note="Before reboot",
                    is_safe=True,
                    idempotency_key="restart-checkin-key",
                ),
            )
            assert replayed_checkin.model_dump(mode="json") == checkin_body
            assert replayed_checkin.late_minutes == 12

        # No duplicated side effects after restart.
        with second_db.session() as session:
            assert session.get(GearInventory, inventory_id).quantity_available == 4
            assert (
                session.scalar(
                    select(func.count())
                    .select_from(InventoryMovement)
                    .where(
                        InventoryMovement.inventory_id == inventory_id,
                        InventoryMovement.movement_type == InventoryMovementType.LOAN_OUT,
                    )
                )
                == 1
            )
            assert session.scalar(select(func.count()).select_from(GearLoan)) == 1
            assert (
                session.scalar(
                    select(func.count())
                    .select_from(AuditLog)
                    .where(
                        AuditLog.entity_type == "itinerary_check_in",
                        AuditLog.correlation_id == "restart-checkin-key",
                    )
                )
                == 1
            )
            assert _business_idem_count(session) == 2
            assert session.scalar(
                select(func.count())
                .select_from(IdempotencyRecord)
                .where(IdempotencyRecord.status == "pending")
            ) == 0
    finally:
        second_db.engine.dispose()


# --------------------------------------------- 5. end-to-end HTTP concurrency + restart


def test_http_concurrent_requests_and_restart_replay(tmp_path) -> None:
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'http.db'}",
        sqlite_timeout_seconds=10,
        sqlite_busy_retries=2,
        sqlite_busy_backoff_seconds=0.01,
    )
    app = create_app(settings)
    with TestClient(app):
        database = app.state.database
        owner, _, inventory_id = _make_inventory(database, quantity=8)
        organizer, _, check_in_id, due = _make_check_in(database)
        with database.session() as session:
            borrower = create_user(session, email="http-borrow@example.com", name="HTTP")
        loaned_at = datetime.now(BEIJING)

        worker_count = 5
        barrier = threading.Barrier(worker_count)

        def http_adjust() -> tuple[int, dict] | Exception:
            local = TestClient(app)
            try:
                barrier.wait()
                response = local.post(
                    f"/api/v1/gear/inventory/{inventory_id}/adjust",
                    json={
                        "quantity_delta": 2,
                        "reason": "HTTP concurrent",
                        "actor_id": owner,
                        "idempotency_key": "http-adjust-key",
                    },
                )
                return response.status_code, response.json()
            except Exception as exc:
                return exc
            finally:
                local.close()

        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            adjust_results = list(pool.map(lambda _: http_adjust(), range(worker_count)))
        assert not any(isinstance(r, Exception) for r in adjust_results), adjust_results
        statuses = {status for status, _ in adjust_results}
        assert statuses == {200}
        bodies = [body for _, body in adjust_results]
        assert all(body == bodies[0] for body in bodies)
        assert bodies[0]["quantity_total"] == 10

        def http_loan() -> tuple[int, dict] | Exception:
            local = TestClient(app)
            try:
                response = local.post(
                    "/api/v1/gear/loans",
                    json={
                        "inventory_id": inventory_id,
                        "borrower_id": borrower,
                        "quantity": 3,
                        "loaned_at": loaned_at.isoformat(),
                        "due_at": (loaned_at + timedelta(days=2)).isoformat(),
                        "actor_id": owner,
                        "idempotency_key": "http-loan-key",
                    },
                )
                return response.status_code, response.json()
            finally:
                local.close()

        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            loan_results = list(pool.map(lambda _: http_loan(), range(worker_count)))
        assert {status for status, _ in loan_results} == {201}
        loan_bodies = [body for _, body in loan_results]
        assert all(body == loan_bodies[0] for body in loan_bodies)

    # Restart the whole application and replay both keys over HTTP.
    app2 = create_app(settings)
    with TestClient(app2) as client2:
        replay_adjust = client2.post(
            f"/api/v1/gear/inventory/{inventory_id}/adjust",
            json={
                "quantity_delta": 2,
                "reason": "HTTP concurrent",
                "actor_id": owner,
                "idempotency_key": "http-adjust-key",
            },
        )
        assert replay_adjust.status_code == 200
        assert replay_adjust.json() == bodies[0]

        replay_loan = client2.post(
            "/api/v1/gear/loans",
            json={
                "inventory_id": inventory_id,
                "borrower_id": borrower,
                "quantity": 3,
                "loaned_at": loaned_at.isoformat(),
                "due_at": (loaned_at + timedelta(days=2)).isoformat(),
                "actor_id": owner,
                "idempotency_key": "http-loan-key",
            },
        )
        assert replay_loan.status_code == 201
        assert replay_loan.json() == loan_bodies[0]

        # Different payload over HTTP must be a stable 409 idempotency conflict.
        clash = client2.post(
            "/api/v1/gear/loans",
            json={
                "inventory_id": inventory_id,
                "borrower_id": borrower,
                "quantity": 1,
                "loaned_at": loaned_at.isoformat(),
                "due_at": (loaned_at + timedelta(days=2)).isoformat(),
                "actor_id": owner,
                "idempotency_key": "http-loan-key",
            },
        )
        assert clash.status_code == 409
        assert clash.json()["detail"]["code"] == "idempotency_conflict"

        database2: Database = app2.state.database
        with database2.session() as session:
            inventory = session.get(GearInventory, inventory_id)
            assert inventory.quantity_total == 10
            assert inventory.quantity_available == 7  # +2 adjustment, 3 loaned out once
            assert session.scalar(select(func.count()).select_from(GearLoan)) == 1
            assert (
                session.scalar(
                    select(func.count())
                    .select_from(InventoryMovement)
                    .where(InventoryMovement.inventory_id == inventory_id)
                )
                == 3  # initial + adjustment + one loan_out
            )
            assert (
                session.scalar(
                    select(func.count())
                    .select_from(AuditLog)
                    .where(AuditLog.correlation_id == "http-loan-key")
                )
                == 1
            )
            assert _business_idem_count(session) == 2


# -------------------------------------------------- 6. registration entry smoke (3rd entry)


def test_concurrent_registration_is_idempotent(database: Database) -> None:
    from trailforge.schemas.activities import RegistrationCreate
    from trailforge.services.activities import ExpeditionService

    with database.session() as session:
        organizer = create_user(session, email="reg-org@example.com", name="Org")
        route = create_route(session, actor_id=organizer)
        expedition_id = create_expedition(session, organizer_id=organizer, route_id=route)
        participant = create_user(session, email="reg-user@example.com", name="Participant")
        from trailforge.schemas.activities import ActivityStateChange

        ExpeditionService(session).change_status(
            expedition_id,
            ActivityStateChange(target_status="open", actor_id=organizer),
        )

    workers = 5
    barrier = threading.Barrier(workers)

    def call() -> object:
        with database.session() as session:
            barrier.wait()
            try:
                return ExpeditionService(session).register(
                    expedition_id,
                    RegistrationCreate(
                        user_id=participant,
                        role="member",
                        idempotency_key="concurrent-register-key",
                    ),
                )
            except Exception as exc:
                return exc

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda _: call(), range(workers)))
    assert not any(isinstance(r, Exception) for r in results), results
    assert all(r.id == results[0].id for r in results)
    with database.session() as session:
        assert (
            session.scalar(
                select(func.count())
                .select_from(ExpeditionRegistration)
                .where(ExpeditionRegistration.expedition_id == expedition_id)
            )
            == 2  # organizer + participant
        )
        assert (
            session.scalar(
                select(func.count())
                .select_from(AuditLog)
                .where(AuditLog.correlation_id == "concurrent-register-key")
            )
            == 1
        )
        records = list(session.scalars(select(IdempotencyRecord)))
        assert len(records) == 1
        assert records[0].status == "completed"


def _return_payload(
    actor_id: int, returned_at: datetime, *, quantity: int, key: str
):
    from trailforge.schemas.gear import GearLoanReturn

    return GearLoanReturn(
        quantity=quantity,
        returned_at=returned_at,
        condition_in="good",
        actor_id=actor_id,
        idempotency_key=key,
    )
