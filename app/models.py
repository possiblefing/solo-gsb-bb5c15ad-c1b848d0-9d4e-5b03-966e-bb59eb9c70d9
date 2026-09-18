"""Pydantic schemas for the batch reconciliation API."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field, StrictInt, StrictStr, field_validator

MAX_TRANSACTIONS_PER_SIDE = 12


class TransactionIn(BaseModel):
    """One fund-flow record.

    - ``id``: batch-unique identifier (non-empty string).
    - ``time``: ISO-8601 timestamp; naive values are treated as UTC.
    - ``amount``: non-zero signed integer amount in cents.
    """

    id: StrictStr = Field(min_length=1, max_length=128)
    time: datetime
    amount: StrictInt

    @field_validator("amount")
    @classmethod
    def amount_must_be_non_zero(cls, v: int) -> int:
        if v == 0:
            raise ValueError("amount must be a non-zero integer (cents)")
        return v

    def epoch_seconds(self) -> float:
        """Return the timestamp as UTC epoch seconds."""
        dt = self.time
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()


class LockedGroupIn(BaseModel):
    """An operator-confirmed group, referencing transactions of this batch.

    ``left_ids`` must reference ids from ``left`` and ``right_ids`` ids from
    ``right``; the referenced transactions must form a legal group. Semantic
    validation (unknown ids, reuse across groups, group legality) happens in
    the endpoint and returns 422 with the offending group's index.
    """

    left_ids: list[StrictStr] = Field(min_length=1, max_length=3)
    right_ids: list[StrictStr] = Field(min_length=1, max_length=3)


class ExcludedPairIn(BaseModel):
    """An operator-declared pair that must never share a reconciliation group.

    ``left_id`` must reference an id from ``left`` and ``right_id`` an id from
    ``right``.  Semantic validation (unknown ids, wrong side, duplicate pairs)
    happens in the endpoint and returns 422 with the offending pair's index.
    """

    left_id: StrictStr = Field(min_length=1, max_length=128)
    right_id: StrictStr = Field(min_length=1, max_length=128)


class BatchRequest(BaseModel):
    """Batch of fund flows from two sides (at most 12 each)."""

    left: list[TransactionIn] = Field(default_factory=list, max_length=MAX_TRANSACTIONS_PER_SIDE)
    right: list[TransactionIn] = Field(default_factory=list, max_length=MAX_TRANSACTIONS_PER_SIDE)
    excluded_pairs: list[ExcludedPairIn] | None = Field(
        default=None,
        description=(
            "Optional (left_id, right_id) pairs that must never end up in the "
            "same reconciliation group."
        ),
    )


class ReconcileRequest(BatchRequest):
    """Reconcile batch, optionally carrying operator constraints."""

    locked_groups: list[LockedGroupIn] | None = Field(
        default=None,
        description="Optional operator-confirmed groups forced into the solution.",
    )


class GroupOut(BaseModel):
    """One reconciliation group: sorted id lists of both sides."""

    left_ids: list[str]
    right_ids: list[str]
    amount: int  # signed reconciled amount in cents (sum of either side)


class ReconcileResponse(BaseModel):
    groups: list[GroupOut]
    total_amount: int  # sum of absolute reconciled amounts across groups
    unmatched_left_ids: list[str]
    unmatched_right_ids: list[str]


class UnmatchedSummaryRequest(ReconcileRequest):
    """Reconcile batch plus the IANA timezone used to localise unmatched flows.

    The batch fields and their validation are identical to
    ``ReconcileRequest``; ``timezone`` must be a valid IANA key such as
    ``"Asia/Shanghai"`` — anything else is rejected with 422.
    """

    timezone: StrictStr = Field(min_length=1, max_length=64)

    @field_validator("timezone")
    @classmethod
    def timezone_must_be_iana_key(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown IANA timezone: {v!r}") from exc
        return v


class SignSummary(BaseModel):
    """Count and absolute amount (cents) of unmatched flows of one sign."""

    count: int = 0
    amount: int = 0  # sum of absolute amounts in cents


class SignBreakdown(BaseModel):
    """Unmatched flows split by amount sign."""

    positive: SignSummary = Field(default_factory=SignSummary)
    negative: SignSummary = Field(default_factory=SignSummary)


class DaySummary(SignBreakdown):
    """Unmatched flows attributed to one local calendar day (YYYY-MM-DD)."""

    date: date


class UnmatchedSummaryResponse(BaseModel):
    timezone: str  # validated IANA key the local dates were computed with
    days: list[DaySummary]  # ascending by local date; only days with unmatched flows
    total: SignBreakdown  # whole-batch totals across all days


class ProposedGroupIn(LockedGroupIn):
    """One group of the externally proposed reconciliation plan under audit.

    Same shape constraints as a locked group (1-3 ids per side).  Semantic
    validation (references, reuse across groups, excluded pairs, legality) is
    reported in the audit response keyed by the group's 0-based index.
    """


class AuditRequest(BatchRequest):
    """Batch plus an externally proposed complete reconciliation plan.

    The batch fields and their validation are identical to
    ``ReconcileRequest`` (duplicate ids -> 400, invalid ``excluded_pairs`` ->
    422); ``proposed_groups`` is the plan under audit and is checked group by
    group, with every failure collected in the response instead of aborting
    at the first one.
    """

    proposed_groups: list[ProposedGroupIn] = Field(
        description="The complete reconciliation plan to audit (may be empty).",
    )


# Machine-readable identifiers of the per-group audit checks.
AuditErrorCode = Literal[
    "unknown_id",      # references a transaction id absent from the batch
    "duplicate_id",    # same id twice inside one group
    "reused",          # transaction already claimed by an earlier proposed group
    "excluded_pair",   # group contains an operator-declared excluded pair
    "shape",           # not one of the five legal group shapes
    "sign",            # amounts inside the group do not share one sign
    "sum_mismatch",    # left and right side sums differ
    "span",            # group spans more than 48 hours
]


class AuditError(BaseModel):
    """One proposed-group validation failure."""

    index: int  # 0-based index into proposed_groups
    code: AuditErrorCode
    msg: str  # human-readable detail, prefixed with the group's index


class AuditResponse(BaseModel):
    valid: bool  # every proposed group passed every check
    errors: list[AuditError]  # empty when valid; ascending by group index
    # The fields below are only computed for a fully legal plan (null otherwise).
    optimal: bool | None  # proposed plan == canonical optimal solution
    proposed_total: int | None  # sum of absolute reconciled amounts of the plan
    optimal_total: int | None  # total of the canonical optimal solution
    # Why a valid plan is not optimal: "suboptimal_amount" (proposed_total <
    # optimal_total) or "non_canonical" (same total, but not the
    # lexicographically canonical grouping); null when optimal or invalid.
    difference: Literal["suboptimal_amount", "non_canonical"] | None
    groups: list[GroupOut] | None  # canonical optimal groups, for correction
