from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from tests.conftest import create_expedition, create_route, create_user
from trailforge.config import Settings
from trailforge.database.migrations import initialize_database
from trailforge.database.session import Database
from trailforge.domain.enums import InventoryMovementType, LoanStatus
from trailforge.errors import IdempotencyConflictError, InventoryError
from trailforge.main import create_app
from trailforge.models.activities import Expedition, ExpeditionRegistration
from trailforge.models.audit import AuditLog, IdempotencyRecord
from trailforge.models.gear import GearInventory, GearLoan, InventoryMovement
from trailforge.models.safety import ItineraryCheckIn
from trailforge.schemas.activities import ActivityStateChange, RegistrationCreate
from trailforge.schemas.gear import (
    GearCatalogCreate,
    GearInventoryCreate,
    GearLoanCreate,
    GearLoanReturn,
    InventoryAdjustment,
)
from trailforge.schemas.safety import CheckInScheduleCreate, CheckInSubmit
from trailforge.services.activities import ExpeditionService
from trailforge.services.gear import GearService
from trailforge.services.safety import SafetyService


def _run_concurrently(
    database: Database,
    attempt: Callable[[Session, int], Any],
    workers: int,
) -> tuple[list[Any], list[Exception]]:
    """Run ``attempt`` in ``workers`` threads, each on its own session/connection."""
    responses: list[Any] = []
    errors: list[Exception] = []

    def worker(index: int) -> None:
        try:
            with database.session() as session:
                responses.append(attempt(session, index))
        except Exception as exc:  # noqa: BLE001 - collected for assertions
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(worker, range(workers)))
    return responses, errors


def _inventory(database: Database, quantity: int = 10) -> tuple[int, int]:
    with database.session() as session:
        owner = create_user(session)
        service = GearService(session)
        catalog = service.create_catalog(
            GearCatalogCreate(sku="TENT-2P", name="Two person tent", category="shelter"),
            actor_id=owner,
        )
        inventory = service.create_inventory(
            GearInventoryCreate(
                catalog_id=catalog.id,
                ownership="club",
                quantity_total=quantity,
                condition="good",
                actor_id=owner,
                idempotency_key="setup-inventory",
            )
        )
        return owner, inventory.id


def _borrower(database: Database) -> int:
    with database.session() as session:
        return create_user(session, email="borrower@example.com", name="Borrower")


def _scheduled_check_in(database: Database) -> tuple[int, int, datetime]:
    with database.session() as session:
        organizer = create_user(session)
        route_id = create_route(session, actor_id=organizer)
        expedition_id = create_expedition(session, organizer_id=organizer, route_id=route_id)
        expedition = session.get(Expedition, expedition_id)
        due = expedition.start_at + timedelta(hours=1)
        check_in = SafetyService(session).schedule_check_in(
            expedition_id,
            CheckInScheduleCreate(user_id=organizer, check_in_type="routine", due_at=due),
            actor_id=organizer,
        )
        return organizer, check_in.id, due


def _open_expedition(database: Database) -> tuple[int, int]:
    with database.session() as session:
        organizer = create_user(session)
        route_id = create_route(session, actor_id=organizer)
        expedition_id = create_expedition(session, organizer_id=organizer, route_id=route_id)
        ExpeditionService(session).change_status(
            expedition_id,
            ActivityStateChange(target_status="open", actor_id=organizer),
        )
        participant = create_user(session, email="participant@example.com", name="Participant")
        return expedition_id, participant


def _movement_count(session: Session, inventory_id: int, movement_type: str) -> int:
    return session.scalar(
        select(func.count())
        .select_from(InventoryMovement)
        .where(
            InventoryMovement.inventory_id == inventory_id,
            InventoryMovement.movement_type == movement_type,
        )
    )


def _audit_count(session: Session, entity_type: str, action: str) -> int:
    return session.scalar(
        select(func.count())
        .select_from(AuditLog)
        .where(AuditLog.entity_type == entity_type, AuditLog.action == action)
    )


def _records(session: Session, scope: str) -> list[IdempotencyRecord]:
    return list(
        session.scalars(select(IdempotencyRecord).where(IdempotencyRecord.scope == scope))
    )


# ---------------------------------------------------------------------------
# Concurrent retries of the same key: one side effect, one committed response
# ---------------------------------------------------------------------------


def test_concurrent_inventory_adjustment_applies_effects_once(database: Database) -> None:
    owner, inventory_id = _inventory(database, quantity=10)
    request = InventoryAdjustment(
        quantity_delta=2, reason="Donation", actor_id=owner, idempotency_key="concurrent-adjust"
    )

    responses, errors = _run_concurrently(
        database, lambda session, _: GearService(session).adjust_inventory(inventory_id, request), 8
    )

    assert errors == []
    assert len(responses) == 8
    assert {response.model_dump_json() for response in responses} == {
        responses[0].model_dump_json()
    }
    assert responses[0].quantity_total == 12
    with database.session() as session:
        inventory = session.get(GearInventory, inventory_id)
        assert inventory.quantity_total == 12
        assert inventory.quantity_available == 12
        assert _movement_count(session, inventory_id, InventoryMovementType.INITIAL) == 1
        assert _movement_count(session, inventory_id, InventoryMovementType.ADJUSTMENT) == 1
        assert _audit_count(session, "gear_inventory", "inventory_changed") == 1
        records = _records(session, f"gear:inventory:{inventory_id}:adjust")
        assert len(records) == 1
        assert records[0].response_json["quantity_total"] == 12


def test_concurrent_inventory_creation_inserts_one_row(database: Database) -> None:
    with database.session() as session:
        owner = create_user(session)
        catalog = GearService(session).create_catalog(
            GearCatalogCreate(sku="CLUB-RAFT", name="Club raft", category="water"),
            actor_id=owner,
        )
        catalog_id = catalog.id

    def attempt(session: Session, _: int) -> Any:
        return GearService(session).create_inventory(
            GearInventoryCreate(
                catalog_id=catalog_id,
                ownership="club",
                quantity_total=4,
                condition="good",
                actor_id=owner,
                idempotency_key="concurrent-create",
            )
        )

    responses, errors = _run_concurrently(database, attempt, 6)

    assert errors == []
    assert len({response.id for response in responses}) == 1
    with database.session() as session:
        inventories = list(session.scalars(select(GearInventory)))
        assert len(inventories) == 1
        assert _movement_count(session, inventories[0].id, InventoryMovementType.INITIAL) == 1
        assert _audit_count(session, "gear_inventory", "created") == 1
        assert len(_records(session, "gear:inventory:create")) == 1


def test_concurrent_loan_same_key_single_side_effect(database: Database) -> None:
    owner, inventory_id = _inventory(database, quantity=3)
    borrower = _borrower(database)
    loaned_at = datetime(2026, 10, 1, 8, 0, tzinfo=UTC)
    request = GearLoanCreate(
        inventory_id=inventory_id,
        borrower_id=borrower,
        quantity=1,
        loaned_at=loaned_at,
        due_at=loaned_at + timedelta(days=3),
        actor_id=owner,
        idempotency_key="concurrent-loan",
    )

    responses, errors = _run_concurrently(
        database, lambda session, _: GearService(session).loan(request), 6
    )

    assert errors == []
    assert len({response.id for response in responses}) == 1
    assert {response.model_dump_json() for response in responses} == {
        responses[0].model_dump_json()
    }
    with database.session() as session:
        assert len(list(session.scalars(select(GearLoan)))) == 1
        assert session.get(GearInventory, inventory_id).quantity_available == 2
        assert _movement_count(session, inventory_id, InventoryMovementType.LOAN_OUT) == 1
        assert _audit_count(session, "gear_loan", "loaned") == 1
        assert len(_records(session, f"gear:inventory:{inventory_id}:loan")) == 1


def test_concurrent_check_in_submit_same_key(database: Database) -> None:
    organizer, check_in_id, due = _scheduled_check_in(database)
    request = CheckInSubmit(
        checked_in_at=due + timedelta(minutes=45),
        is_safe=True,
        note="Delayed by terrain",
        idempotency_key="concurrent-checkin",
    )

    responses, errors = _run_concurrently(
        database,
        lambda session, _: SafetyService(session).submit_check_in(check_in_id, request),
        6,
    )

    assert errors == []
    assert {response.model_dump_json() for response in responses} == {
        responses[0].model_dump_json()
    }
    assert responses[0].late_minutes == 45
    with database.session() as session:
        check_in = session.get(ItineraryCheckIn, check_in_id)
        assert check_in.checked_in_at == due + timedelta(minutes=45)
        assert check_in.late_minutes == 45
        assert _audit_count(session, "itinerary_check_in", "checked_in") == 1
        assert len(_records(session, f"safety:check-in:{check_in_id}:submit")) == 1


def test_concurrent_registration_same_key(database: Database) -> None:
    expedition_id, participant = _open_expedition(database)
    request = RegistrationCreate(user_id=participant, idempotency_key="concurrent-register")

    responses, errors = _run_concurrently(
        database,
        lambda session, _: ExpeditionService(session).register(expedition_id, request),
        6,
    )

    assert errors == []
    assert len({response.id for response in responses}) == 1
    with database.session() as session:
        registrations = list(
            session.scalars(
                select(ExpeditionRegistration).where(
                    ExpeditionRegistration.expedition_id == expedition_id,
                    ExpeditionRegistration.user_id == participant,
                )
            )
        )
        assert len(registrations) == 1
        assert _audit_count(session, "expedition_registration", "registered") == 1
        assert len(_records(session, f"expedition:{expedition_id}:register")) == 1


def test_concurrent_same_key_different_payload_has_single_winner(database: Database) -> None:
    owner, inventory_id = _inventory(database, quantity=10)

    def attempt(session: Session, index: int) -> Any:
        delta = 2 if index % 2 == 0 else 3
        return GearService(session).adjust_inventory(
            inventory_id,
            InventoryAdjustment(
                quantity_delta=delta,
                reason=f"Batch {delta}",
                actor_id=owner,
                idempotency_key="mixed-payload-key",
            ),
        )

    responses, errors = _run_concurrently(database, attempt, 8)

    assert len(responses) + len(errors) == 8
    assert len({response.model_dump_json() for response in responses}) == 1
    winning_delta = responses[0].quantity_total - 10
    assert winning_delta in {2, 3}
    assert errors, "the losing payload must conflict, never execute"
    assert all(isinstance(error, IdempotencyConflictError) for error in errors)
    with database.session() as session:
        assert session.get(GearInventory, inventory_id).quantity_total == 10 + winning_delta
        assert _movement_count(session, inventory_id, InventoryMovementType.ADJUSTMENT) == 1
        assert _audit_count(session, "gear_inventory", "inventory_changed") == 1
        assert len(_records(session, f"gear:inventory:{inventory_id}:adjust")) == 1


# ---------------------------------------------------------------------------
# Rollback leaves no placeholder; the key stays usable
# ---------------------------------------------------------------------------


def test_failed_attempt_leaves_no_placeholder_and_key_can_be_retried(
    database: Database,
) -> None:
    owner, inventory_id = _inventory(database, quantity=5)
    scope = f"gear:inventory:{inventory_id}:adjust"
    with pytest.raises(InventoryError, match="negative"), database.session() as session:
        GearService(session).adjust_inventory(
            inventory_id,
            InventoryAdjustment(
                quantity_delta=-99,
                reason="Broken count",
                actor_id=owner,
                idempotency_key="retry-after-failure",
            ),
        )
    with database.session() as session:
        assert _records(session, scope) == []
        assert _movement_count(session, inventory_id, InventoryMovementType.ADJUSTMENT) == 0
        assert _audit_count(session, "gear_inventory", "inventory_changed") == 0
        assert session.get(GearInventory, inventory_id).quantity_total == 5

    # The rolled-back attempt committed no record, so the same key may be
    # reused immediately, even with a corrected (different) payload.
    with database.session() as session:
        response = GearService(session).adjust_inventory(
            inventory_id,
            InventoryAdjustment(
                quantity_delta=-1,
                reason="Corrected count",
                actor_id=owner,
                idempotency_key="retry-after-failure",
            ),
        )
    assert response.quantity_total == 4
    with database.session() as session:
        assert session.get(GearInventory, inventory_id).quantity_total == 4
        assert _movement_count(session, inventory_id, InventoryMovementType.ADJUSTMENT) == 1
        assert _audit_count(session, "gear_inventory", "inventory_changed") == 1
        assert len(_records(session, scope)) == 1


# ---------------------------------------------------------------------------
# Replay returns the first committed response, not the mutated current state
# ---------------------------------------------------------------------------


def test_replay_returns_original_response_after_later_updates(database: Database) -> None:
    owner, inventory_id = _inventory(database, quantity=10)
    request = InventoryAdjustment(
        quantity_delta=2, reason="Donation", actor_id=owner, idempotency_key="snapshot-adjust"
    )
    with database.session() as session:
        first = GearService(session).adjust_inventory(inventory_id, request)
    assert first.quantity_total == 12
    with database.session() as session:
        GearService(session).adjust_inventory(
            inventory_id,
            InventoryAdjustment(
                quantity_delta=5,
                reason="Later correction",
                actor_id=owner,
                idempotency_key="later-adjustment",
            ),
        )
    with database.session() as session:
        replayed = GearService(session).adjust_inventory(inventory_id, request)

    assert replayed.model_dump() == first.model_dump()
    assert replayed.quantity_total == 12
    with database.session() as session:
        assert session.get(GearInventory, inventory_id).quantity_total == 17
        # The replay itself wrote no extra movement, audit, or record.
        assert _movement_count(session, inventory_id, InventoryMovementType.ADJUSTMENT) == 2
        assert _audit_count(session, "gear_inventory", "inventory_changed") == 2
        assert len(_records(session, f"gear:inventory:{inventory_id}:adjust")) == 2


def test_loan_replay_restores_typed_snapshot_after_return(database: Database) -> None:
    owner, inventory_id = _inventory(database, quantity=2)
    borrower = _borrower(database)
    loaned_at = datetime(2026, 10, 1, 8, 0, tzinfo=UTC)
    request = GearLoanCreate(
        inventory_id=inventory_id,
        borrower_id=borrower,
        quantity=1,
        loaned_at=loaned_at,
        due_at=loaned_at + timedelta(days=3),
        actor_id=owner,
        idempotency_key="loan-snapshot",
    )
    with database.session() as session:
        first = GearService(session).loan(request)
    with database.session() as session:
        GearService(session).return_loan(
            first.id,
            GearLoanReturn(
                quantity=1,
                returned_at=loaned_at + timedelta(days=1),
                condition_in="fair",
                actor_id=owner,
                idempotency_key="loan-return-later",
            ),
        )
    with database.session() as session:
        replayed = GearService(session).loan(request)

    assert replayed.model_dump() == first.model_dump()
    assert replayed.status == LoanStatus.ACTIVE
    assert isinstance(replayed.loaned_at, datetime)
    assert replayed.loaned_at.tzinfo is not None
    with database.session() as session:
        current = session.get(GearLoan, first.id)
        assert current.status == LoanStatus.RETURNED
        assert _audit_count(session, "gear_loan", "loaned") == 1


# ---------------------------------------------------------------------------
# Canonical hashing: timezone and enum representations must not drift
# ---------------------------------------------------------------------------


def test_timezone_equivalent_payload_replays_instead_of_conflicting(
    database: Database,
) -> None:
    owner, inventory_id = _inventory(database, quantity=2)
    borrower = _borrower(database)
    shanghai = timezone(timedelta(hours=8))
    with database.session() as session:
        first = GearService(session).loan(
            GearLoanCreate(
                inventory_id=inventory_id,
                borrower_id=borrower,
                quantity=1,
                loaned_at=datetime(2026, 10, 1, 10, 0, tzinfo=shanghai),
                due_at=datetime(2026, 10, 4, 10, 0, tzinfo=shanghai),
                actor_id=owner,
                idempotency_key="tz-loan-key",
            )
        )
    # Same instants expressed in UTC: the same request, not a conflict.
    with database.session() as session:
        replayed = GearService(session).loan(
            GearLoanCreate(
                inventory_id=inventory_id,
                borrower_id=borrower,
                quantity=1,
                loaned_at=datetime(2026, 10, 1, 2, 0, tzinfo=UTC),
                due_at=datetime(2026, 10, 4, 2, 0, tzinfo=UTC),
                actor_id=owner,
                idempotency_key="tz-loan-key",
            )
        )
    assert replayed.model_dump() == first.model_dump()
    with database.session() as session:
        assert len(list(session.scalars(select(GearLoan)))) == 1
    # A genuinely different instant with the same key is a stable conflict.
    with pytest.raises(IdempotencyConflictError), database.session() as session:
        GearService(session).loan(
            GearLoanCreate(
                inventory_id=inventory_id,
                borrower_id=borrower,
                quantity=1,
                loaned_at=datetime(2026, 10, 1, 3, 0, tzinfo=UTC),
                due_at=datetime(2026, 10, 4, 2, 0, tzinfo=UTC),
                actor_id=owner,
                idempotency_key="tz-loan-key",
            )
        )
    with database.session() as session:
        assert len(list(session.scalars(select(GearLoan)))) == 1
        assert session.get(GearInventory, inventory_id).quantity_available == 1


def test_enum_difference_in_payload_is_a_stable_conflict(database: Database) -> None:
    owner, inventory_id = _inventory(database, quantity=2)
    borrower = _borrower(database)
    loaned_at = datetime(2026, 10, 1, 8, 0, tzinfo=UTC)
    with database.session() as session:
        loan = GearService(session).loan(
            GearLoanCreate(
                inventory_id=inventory_id,
                borrower_id=borrower,
                quantity=1,
                loaned_at=loaned_at,
                due_at=loaned_at + timedelta(days=3),
                actor_id=owner,
                idempotency_key="enum-loan-setup",
            )
        )
    returned_at = loaned_at + timedelta(days=1)
    with database.session() as session:
        GearService(session).return_loan(
            loan.id,
            GearLoanReturn(
                quantity=1,
                returned_at=returned_at,
                condition_in="good",
                actor_id=owner,
                idempotency_key="enum-return-key",
            ),
        )
    with pytest.raises(IdempotencyConflictError), database.session() as session:
        GearService(session).return_loan(
            loan.id,
            GearLoanReturn(
                quantity=1,
                returned_at=returned_at,
                condition_in="fair",
                actor_id=owner,
                idempotency_key="enum-return-key",
            ),
        )
    with database.session() as session:
        assert _movement_count(session, inventory_id, InventoryMovementType.RETURN_IN) == 1
        assert _audit_count(session, "gear_loan", "returned") == 1


# ---------------------------------------------------------------------------
# Idempotency records survive an application restart
# ---------------------------------------------------------------------------


def test_idempotency_survives_application_restart(settings: Settings) -> None:
    first_db = Database(settings)
    initialize_database(first_db)
    with first_db.session() as session:
        owner = create_user(session)
        route_id = create_route(session, actor_id=owner)
        expedition_id = create_expedition(session, organizer_id=owner, route_id=route_id)
        expedition = session.get(Expedition, expedition_id)
        due = expedition.start_at + timedelta(hours=1)
        gear = GearService(session)
        catalog = gear.create_catalog(
            GearCatalogCreate(sku="TENT-2P", name="Two person tent", category="shelter"),
            actor_id=owner,
        )
        inventory = gear.create_inventory(
            GearInventoryCreate(
                catalog_id=catalog.id,
                ownership="club",
                quantity_total=5,
                condition="good",
                actor_id=owner,
                idempotency_key="restart-inventory",
            )
        )
        check_in = SafetyService(session).schedule_check_in(
            expedition_id,
            CheckInScheduleCreate(user_id=owner, check_in_type="routine", due_at=due),
            actor_id=owner,
        )
        inventory_id = inventory.id
        check_in_id = check_in.id
    with first_db.session() as session:
        adjust_request = InventoryAdjustment(
            quantity_delta=2, reason="Donation", actor_id=owner, idempotency_key="restart-adjust"
        )
        first_adjust = GearService(session).adjust_inventory(inventory_id, adjust_request)
    loaned_at = datetime(2026, 10, 1, 8, 0, tzinfo=UTC)
    with first_db.session() as session:
        loan_request = GearLoanCreate(
            inventory_id=inventory_id,
            borrower_id=owner,
            quantity=1,
            loaned_at=loaned_at,
            due_at=loaned_at + timedelta(days=3),
            actor_id=owner,
            idempotency_key="restart-loan",
        )
        first_loan = GearService(session).loan(loan_request)
    with first_db.session() as session:
        submit_request = CheckInSubmit(
            checked_in_at=due + timedelta(minutes=10),
            is_safe=True,
            note="All good",
            idempotency_key="restart-checkin",
        )
        first_check_in = SafetyService(session).submit_check_in(check_in_id, submit_request)
    first_db.engine.dispose()

    restarted = Database(settings)
    initialize_database(restarted)
    with restarted.session() as session:
        replayed_adjust = GearService(session).adjust_inventory(inventory_id, adjust_request)
        assert replayed_adjust.model_dump() == first_adjust.model_dump()
        assert replayed_adjust.quantity_total == 7
    with restarted.session() as session:
        replayed_loan = GearService(session).loan(loan_request)
        assert replayed_loan.model_dump() == first_loan.model_dump()
        assert replayed_loan.status == LoanStatus.ACTIVE
    with restarted.session() as session:
        replayed_check_in = SafetyService(session).submit_check_in(check_in_id, submit_request)
        assert replayed_check_in.model_dump() == first_check_in.model_dump()
        assert replayed_check_in.late_minutes == 10
    with pytest.raises(IdempotencyConflictError), restarted.session() as session:
        GearService(session).adjust_inventory(
            inventory_id,
            InventoryAdjustment(
                quantity_delta=3,
                reason="Changed payload",
                actor_id=owner,
                idempotency_key="restart-adjust",
            ),
        )
    with restarted.session() as session:
        assert session.get(GearInventory, inventory_id).quantity_total == 7
        assert session.get(GearInventory, inventory_id).quantity_available == 6
        assert _movement_count(session, inventory_id, InventoryMovementType.ADJUSTMENT) == 1
        assert _movement_count(session, inventory_id, InventoryMovementType.LOAN_OUT) == 1
        assert len(list(session.scalars(select(GearLoan)))) == 1
        assert _audit_count(session, "gear_inventory", "inventory_changed") == 1
        assert _audit_count(session, "gear_loan", "loaned") == 1
        assert _audit_count(session, "itinerary_check_in", "checked_in") == 1
        assert len(_records(session, f"gear:inventory:{inventory_id}:adjust")) == 1
        assert len(_records(session, f"gear:inventory:{inventory_id}:loan")) == 1
        assert len(_records(session, f"safety:check-in:{check_in_id}:submit")) == 1
    restarted.engine.dispose()


# ---------------------------------------------------------------------------
# API level: concurrent retries and exact replay over HTTP
# ---------------------------------------------------------------------------


def _setup_inventory_api(client: TestClient) -> tuple[int, int]:
    user = client.post(
        "/api/v1/users", json={"email": "api@example.com", "display_name": "API User"}
    ).json()
    catalog = client.post(
        "/api/v1/gear/catalog",
        params={"actor_id": user["id"]},
        json={"sku": "TENT-2P", "name": "Two person tent", "category": "shelter"},
    ).json()
    inventory = client.post(
        "/api/v1/gear/inventory",
        json={
            "catalog_id": catalog["id"],
            "ownership": "club",
            "quantity_total": 10,
            "condition": "good",
            "actor_id": user["id"],
            "idempotency_key": "api-setup-inventory",
        },
    ).json()
    return user["id"], inventory["id"]


def test_api_concurrent_retries_single_side_effect(client: TestClient) -> None:
    actor_id, inventory_id = _setup_inventory_api(client)
    payload = {
        "quantity_delta": 2,
        "reason": "Donation",
        "actor_id": actor_id,
        "idempotency_key": "api-concurrent-adjust",
    }
    bodies: list[dict] = []
    failures: list[tuple[int, str]] = []

    def call(_: int) -> None:
        response = client.post(f"/api/v1/gear/inventory/{inventory_id}/adjust", json=payload)
        if response.status_code == 200:
            bodies.append(response.json())
        else:
            failures.append((response.status_code, response.text))

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(call, range(6)))

    assert failures == []
    assert len(bodies) == 6
    assert {json.dumps(body, sort_keys=True) for body in bodies} == {
        json.dumps(bodies[0], sort_keys=True)
    }
    assert bodies[0]["quantity_total"] == 12
    with client.app.state.database.session() as session:
        assert session.get(GearInventory, inventory_id).quantity_total == 12
        assert _movement_count(session, inventory_id, InventoryMovementType.ADJUSTMENT) == 1
        assert _audit_count(session, "gear_inventory", "inventory_changed") == 1
        assert len(_records(session, f"gear:inventory:{inventory_id}:adjust")) == 1


def test_api_replay_matches_first_response_and_writes_nothing(client: TestClient) -> None:
    actor_id, inventory_id = _setup_inventory_api(client)
    payload = {
        "quantity_delta": 2,
        "reason": "Donation",
        "actor_id": actor_id,
        "idempotency_key": "api-replay-adjust",
    }
    first = client.post(f"/api/v1/gear/inventory/{inventory_id}/adjust", json=payload)
    assert first.status_code == 200
    later = client.post(
        f"/api/v1/gear/inventory/{inventory_id}/adjust",
        json={**payload, "quantity_delta": 5, "idempotency_key": "api-later-adjust"},
    )
    assert later.status_code == 200

    replay = client.post(f"/api/v1/gear/inventory/{inventory_id}/adjust", json=payload)
    assert replay.status_code == 200
    assert replay.json() == first.json()
    assert replay.json()["quantity_total"] == 12

    conflict = client.post(
        f"/api/v1/gear/inventory/{inventory_id}/adjust",
        json={**payload, "quantity_delta": 3},
    )
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "idempotency_conflict"

    with client.app.state.database.session() as session:
        assert session.get(GearInventory, inventory_id).quantity_total == 17
        assert _movement_count(session, inventory_id, InventoryMovementType.ADJUSTMENT) == 2
        assert _audit_count(session, "gear_inventory", "inventory_changed") == 2
        assert len(_records(session, f"gear:inventory:{inventory_id}:adjust")) == 2


def test_api_idempotency_across_app_restart(settings: Settings) -> None:
    first_app = create_app(settings)
    with TestClient(first_app) as client:
        actor_id, inventory_id = _setup_inventory_api(client)
        payload = {
            "quantity_delta": 2,
            "reason": "Donation",
            "actor_id": actor_id,
            "idempotency_key": "api-restart-adjust",
        }
        first = client.post(f"/api/v1/gear/inventory/{inventory_id}/adjust", json=payload)
        assert first.status_code == 200

    restarted_app = create_app(settings)
    with TestClient(restarted_app) as client:
        replay = client.post(f"/api/v1/gear/inventory/{inventory_id}/adjust", json=payload)
        assert replay.status_code == 200
        assert replay.json() == first.json()
        conflict = client.post(
            f"/api/v1/gear/inventory/{inventory_id}/adjust",
            json={**payload, "quantity_delta": 3},
        )
        assert conflict.status_code == 409
        assert conflict.json()["detail"]["code"] == "idempotency_conflict"
        with restarted_app.state.database.session() as session:
            assert session.get(GearInventory, inventory_id).quantity_total == 12
            assert _movement_count(session, inventory_id, InventoryMovementType.ADJUSTMENT) == 1
            assert _audit_count(session, "gear_inventory", "inventory_changed") == 1
            assert len(_records(session, f"gear:inventory:{inventory_id}:adjust")) == 1
