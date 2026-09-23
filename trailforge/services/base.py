from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from trailforge.database.base import utc_now
from trailforge.domain.enums import AuditAction
from trailforge.errors import ConflictError, DatabaseBusyError, IdempotencyConflictError
from trailforge.models.audit import AuditLog, IdempotencyRecord
from trailforge.repositories.audit import IdempotencyRepository

SENSITIVE_FIELDS = {
    "password",
    "password_hash",
    "secret",
    "token",
    "api_key",
    "authorization",
}

IDEMPOTENCY_PENDING = "pending"
IDEMPOTENCY_COMPLETED = "completed"


@dataclass(frozen=True)
class IdempotencyLease:
    """Outcome of reserving an idempotency key.

    ``record_id`` is set when the caller owns the reservation and must perform
    the side effects exactly once, then call :meth:`complete`. ``replay`` is set
    when an earlier, already-committed request owns the key: the caller must not
    perform any side effect and must return the stored response verbatim.
    """

    record_id: int | None
    replay: IdempotencyResult | None

    @property
    def is_replay(self) -> bool:
        return self.replay is not None

    @classmethod
    def execute(cls, record_id: int) -> IdempotencyLease:
        return cls(record_id=record_id, replay=None)

    @classmethod
    def completed(cls, result: IdempotencyResult) -> IdempotencyLease:
        return cls(record_id=None, replay=result)


@dataclass(frozen=True)
class IdempotencyResult:
    """Detached view of a stored idempotency response, safe across sessions."""

    record_id: int
    request_hash: str
    resource_type: str
    resource_id: int | None
    status: str
    response_json: dict[str, Any]

    @classmethod
    def from_record(cls, record: IdempotencyRecord) -> IdempotencyResult:
        return cls(
            record_id=record.id,
            request_hash=record.request_hash,
            resource_type=record.resource_type,
            resource_id=record.resource_id,
            status=record.status,
            response_json=dict(record.response_json or {}),
        )


class ServiceBase:
    def __init__(self, session: Session) -> None:
        self.session = session

    def audit(
        self,
        *,
        actor_id: int | None,
        entity_type: str,
        entity_id: int,
        action: AuditAction,
        before: dict[str, Any] | None = None,
        after: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> AuditLog:
        log = AuditLog(
            actor_id=actor_id,
            entity_type=entity_type,
            entity_id=entity_id,
            action=action,
            before_state=self._sanitize(before or {}),
            after_state=self._sanitize(after or {}),
            context=self._sanitize(context or {}),
            correlation_id=correlation_id,
        )
        self.session.add(log)
        self.session.flush()
        return log

    def snapshot(self, entity: object, *fields: str) -> dict[str, Any]:
        if not fields:
            mapper = inspect(entity).mapper
            fields = tuple(column.key for column in mapper.column_attrs)
        values: dict[str, Any] = {}
        for field in fields:
            if field.lower() in SENSITIVE_FIELDS:
                continue
            values[field] = self._json_value(getattr(entity, field, None))
        return values

    def request_hash(self, payload: BaseModel | dict[str, Any]) -> str:
        if isinstance(payload, BaseModel):
            raw = payload.model_dump(mode="python", exclude={"idempotency_key"})
        else:
            raw = payload
        encoded = json.dumps(
            self._canonical(raw),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def begin_idempotent(
        self,
        *,
        scope: str,
        key: str,
        payload: BaseModel | dict[str, Any],
    ) -> IdempotencyLease:
        """Reserve ``(scope, key)`` before any side effect runs.

        The placeholder row is written inside the caller's transaction, so it is
        only visible to other connections once the whole request commits. A
        rolled-back request takes the placeholder with it, leaving the key free.

        On a unique-constraint collision the savepoint is discarded (the business
        transaction stays usable) and the existing row decides the outcome:
        a completed row with the same hash replays, a different hash conflicts,
        and a stale ``pending`` row (only possible if a previous process died
        after committing a reservation outside this flow) is safely reclaimed.
        """
        digest = self.request_hash(payload)
        record = IdempotencyRecord(
            scope=scope,
            idempotency_key=key,
            request_hash=digest,
            resource_type="",
            resource_id=None,
            response_json={},
            status=IDEMPOTENCY_PENDING,
            locked_at=utc_now(),
        )
        try:
            with self.session.begin_nested():
                self.session.add(record)
                self.session.flush()
        except IntegrityError as exc:
            existing = IdempotencyRepository(self.session).get_key(scope, key)
            if existing is None:
                # The insert failed against a row another connection just
                # committed; this transaction's read snapshot may still predate
                # that commit, so re-read on a fresh connection that sees it.
                detached = self._get_committed_key(scope, key)
                if detached is None:
                    raise DatabaseBusyError(
                        "idempotency key is still being processed by another request",
                        context={"scope": scope, "key": key},
                    ) from exc
                if detached.request_hash != digest:
                    raise IdempotencyConflictError(
                        "idempotency key was already used with a different request",
                        context={"scope": scope, "key": key},
                    ) from exc
                return IdempotencyLease.completed(detached)
            if existing.status == IDEMPOTENCY_COMPLETED:
                if existing.request_hash != digest:
                    raise IdempotencyConflictError(
                        "idempotency key was already used with a different request",
                        context={"scope": scope, "key": key},
                    ) from exc
                return IdempotencyLease.completed(IdempotencyResult.from_record(existing))
            # Visible pending row whose owning transaction already ended (an
            # active owner would hold the row invisible or still own the unique
            # key): reclaim it instead of leaving a dead placeholder.
            existing.request_hash = digest
            existing.status = IDEMPOTENCY_PENDING
            existing.locked_at = utc_now()
            existing.resource_type = ""
            existing.resource_id = None
            existing.response_json = {}
            existing.error = None
            self.session.flush()
            return IdempotencyLease.execute(existing.id)
        return IdempotencyLease.execute(record.id)

    def complete_idempotent(
        self,
        lease: IdempotencyLease,
        *,
        resource_type: str,
        resource_id: int,
        response: BaseModel | dict[str, Any],
    ) -> IdempotencyRecord:
        """Freeze the first committed response onto the reserved row."""
        if lease.record_id is None:
            raise ConflictError("cannot complete an idempotency replay")
        record = self.session.get(IdempotencyRecord, lease.record_id)
        if record is None:
            raise ConflictError("idempotency reservation vanished before completion")
        if record.status == IDEMPOTENCY_COMPLETED:
            return record
        if isinstance(response, BaseModel):
            stored: dict[str, Any] = response.model_dump(mode="json")
        else:
            stored = response
        record.resource_type = resource_type
        record.resource_id = resource_id
        record.response_json = self._sanitize(stored)
        record.status = IDEMPOTENCY_COMPLETED
        record.locked_at = None
        record.error = None
        self.session.flush()
        return record

    @staticmethod
    def restore_response(response_type: type, result: IdempotencyResult) -> Any:
        """Rebuild the original response object from its frozen snapshot."""
        return response_type.model_validate(result.response_json)

    def _get_committed_key(
        self,
        scope: str,
        key: str,
        *,
        wait_seconds: float = 5.0,
    ) -> IdempotencyResult | None:
        """Read a completed reservation outside this (poisoned) transaction.

        Polls briefly so a concurrent request that lost the insert race waits for
        the winner to commit and then returns its frozen response instead of
        failing with a constraint error. Rows still ``pending`` at the deadline
        belong to an owner that is alive elsewhere (placeholder and completion
        are committed atomically by this code base), so they are reported as a
        retryable conflict; orphaned rows left by a rolled-back transaction are
        removed by that rollback and never reach here.
        """
        factory = sessionmaker(bind=self.session.bind, future=True)
        deadline = time.monotonic() + wait_seconds
        delay = 0.005
        while True:
            other = factory()
            try:
                row = other.scalar(
                    select(IdempotencyRecord).where(
                        IdempotencyRecord.scope == scope,
                        IdempotencyRecord.idempotency_key == key,
                    )
                )
                if row is not None and row.status == IDEMPOTENCY_COMPLETED:
                    return IdempotencyResult.from_record(row)
            finally:
                other.close()
            if time.monotonic() >= deadline:
                return None
            time.sleep(delay)
            delay = min(delay * 1.5, 0.1)

    @classmethod
    def _canonical(cls, value: Any) -> Any:
        """Normalize payloads so equivalent values hash identically.

        Aware datetimes expressed in different offsets (``+08:00`` vs ``Z``) are
        the same instant, and enums must match whether they arrive as members or
        as their serialized value.
        """
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, datetime):
            if value.tzinfo is None or value.utcoffset() is None:
                return value.isoformat()
            return value.astimezone(UTC).isoformat(timespec="microseconds")
        if isinstance(value, date):
            return value.isoformat()
        if isinstance(value, dict):
            return {str(key): cls._canonical(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._canonical(item) for item in value]
        return value

    @classmethod
    def _sanitize(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): "[REDACTED]"
                if str(key).lower() in SENSITIVE_FIELDS
                else cls._sanitize(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [cls._sanitize(item) for item in value]
        return cls._json_value(value)

    @staticmethod
    def _json_value(value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        if hasattr(value, "isoformat"):
            return value.isoformat()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)
