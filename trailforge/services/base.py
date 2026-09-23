from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from datetime import UTC, date, datetime
from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from trailforge.database.session import is_sqlite_busy_error
from trailforge.domain.enums import AuditAction
from trailforge.errors import DatabaseBusyError, IdempotencyConflictError, TrailForgeError
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

ResponseT = TypeVar("ResponseT", bound=BaseModel)

# Bounded retries for a request that loses the race on the shared
# (scope, idempotency_key) unique constraint, or hits a transient SQLite
# busy/locked error while another connection commits. After a loss the
# request rolls its own side effects back and re-checks the record: the
# winner's committed response is then replayed instead of erroring out.
IDEMPOTENCY_RACE_ATTEMPTS = 4
IDEMPOTENCY_RACE_BACKOFF_SECONDS = 0.05


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

    def execute_idempotent(
        self,
        *,
        scope: str,
        key: str,
        payload: BaseModel | dict[str, Any],
        response_type: type[ResponseT],
        operation: Callable[[], ResponseT],
    ) -> ResponseT:
        """Run ``operation`` at most once for ``(scope, key)``.

        - A committed record with the same payload hash replays the stored
          response snapshot, restored as ``response_type``; no side effect,
          audit entry, or inventory movement is written again.
        - A committed record with a different hash raises
          ``IdempotencyConflictError``; this is stable across retries.
        - Concurrent callers race on the ``(scope, key)`` unique constraint.
          Only one transaction commits its side effects; losers roll back and
          replay the winner's committed response instead of surfacing a
          unique-constraint or locking error.
        - If a racing request still observes the winner's committed side
          effects inside ``operation`` (its snapshot predates the winner's
          commit) it can raise a business error (duplicate loan, already
          submitted, ...). Because the winner's record commits in the same
          transaction as its side effects, re-checking the key after rolling
          back distinguishes a lost race (record now exists: replay it) from
          a genuine business conflict (no record: re-raise).
        - A record is only written together with the side effects in the same
          transaction, so a rolled-back attempt leaves no placeholder behind
          and the key can be retried safely.
        """
        last_error: IntegrityError | OperationalError | None = None
        for attempt in range(IDEMPOTENCY_RACE_ATTEMPTS):
            prior = self.find_idempotent(scope=scope, key=key, payload=payload)
            if prior is not None:
                return self.restore_response(prior, response_type)
            try:
                return operation()
            except (IntegrityError, OperationalError) as exc:
                # Discard every side effect of this attempt before either
                # replaying the winner's response or re-executing cleanly.
                self.session.rollback()
                if not _is_race_error(exc):
                    raise
                last_error = exc
                if attempt < IDEMPOTENCY_RACE_ATTEMPTS - 1:
                    time.sleep(IDEMPOTENCY_RACE_BACKOFF_SECONDS * (2**attempt))
            except TrailForgeError:
                self.session.rollback()
                # A concurrent request with the same key may have committed
                # the side effects that made this attempt fail; its committed
                # record decides whether to replay, conflict, or re-raise.
                prior = self.find_idempotent(scope=scope, key=key, payload=payload)
                if prior is not None:
                    return self.restore_response(prior, response_type)
                raise
        raise DatabaseBusyError(
            "idempotent operation could not be completed while a concurrent "
            "request held the same key",
            context={"scope": scope, "attempts": IDEMPOTENCY_RACE_ATTEMPTS},
        ) from last_error

    @staticmethod
    def restore_response(
        record: IdempotencyRecord, response_type: type[ResponseT]
    ) -> ResponseT:
        """Rebuild the stored response snapshot with the endpoint's schema type.

        Validating the stored JSON through ``response_type`` restores
        timezone-aware datetimes and enums exactly as the first response
        produced them, instead of re-reading the resource's current state.
        """
        return response_type.model_validate(record.response_json)

    def request_hash(self, payload: BaseModel | dict[str, Any]) -> str:
        canonical = self._canonical_payload(payload)
        encoded = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def find_idempotent(
        self,
        *,
        scope: str,
        key: str,
        payload: BaseModel | dict[str, Any],
    ) -> IdempotencyRecord | None:
        existing = IdempotencyRepository(self.session).get_key(scope, key)
        if existing is None:
            return None
        request_hash = self.request_hash(payload)
        if existing.request_hash != request_hash:
            raise IdempotencyConflictError(
                "idempotency key was already used with a different request",
                context={"scope": scope, "key": key},
            )
        return existing

    def save_idempotent(
        self,
        *,
        scope: str,
        key: str,
        payload: BaseModel | dict[str, Any],
        resource_type: str,
        resource_id: int,
        response: dict[str, Any],
    ) -> IdempotencyRecord:
        record = IdempotencyRecord(
            scope=scope,
            idempotency_key=key,
            request_hash=self.request_hash(payload),
            resource_type=resource_type,
            resource_id=resource_id,
            response_json=self._sanitize(response),
        )
        self.session.add(record)
        self.session.flush()
        return record

    @classmethod
    def _canonical_payload(cls, payload: BaseModel | dict[str, Any]) -> Any:
        if isinstance(payload, BaseModel):
            raw: Any = payload.model_dump(mode="python", exclude={"idempotency_key"})
        else:
            raw = payload
        return cls._canonical_value(raw)

    @classmethod
    def _canonical_value(cls, value: Any) -> Any:
        """Normalize values so equivalent payloads hash identically.

        Timezone-aware datetimes are reduced to UTC ISO-8601 and enums to
        their values, so serialization drift (offset notation, enum
        representation) cannot turn an identical retry into a false
        conflict, nor a changed payload into a false hit.
        """
        if isinstance(value, BaseModel):
            return cls._canonical_value(value.model_dump(mode="python"))
        if isinstance(value, dict):
            return {str(key): cls._canonical_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._canonical_value(item) for item in value]
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, datetime):
            if value.tzinfo is not None and value.utcoffset() is not None:
                return value.astimezone(UTC).isoformat()
            return value.isoformat()
        if isinstance(value, date):
            return value.isoformat()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

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


def _is_race_error(exc: IntegrityError | OperationalError) -> bool:
    """True if the error comes from losing the idempotency race, not business logic."""
    if isinstance(exc, OperationalError):
        return is_sqlite_busy_error(exc)
    # Only the idempotency table's unique constraint marks a lost race; any
    # other integrity error is a genuine business constraint violation.
    return "idempotency_records" in str(exc).lower()
