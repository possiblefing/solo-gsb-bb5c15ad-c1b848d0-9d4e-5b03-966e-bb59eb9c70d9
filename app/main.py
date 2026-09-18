"""Batch fund-flow reconciliation API."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import FastAPI, Header, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response

from .idempotency import FingerprintConflict, canonical_fingerprint, store
from .models import (
    AuditError,
    AuditRequest,
    AuditResponse,
    BatchRequest,
    DaySummary,
    GroupOut,
    ReconcileRequest,
    ReconcileResponse,
    SignBreakdown,
    UnmatchedSummaryRequest,
    UnmatchedSummaryResponse,
)
from .solver import (
    ExcludedPair,
    ExcludedPairError,
    LockedGroup,
    LockedGroupError,
    Reconciler,
    Txn,
    audit_proposed_groups,
    resolve_excluded_pairs,
    resolve_locked_groups,
)

IDEMPOTENCY_KEY_MIN = 1
IDEMPOTENCY_KEY_MAX = 128


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    store.close()


app = FastAPI(
    title="Batch Reconciliation API",
    version="1.0.0",
    summary="Match fund flows from two sides into optimal reconciliation groups.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def build_reconciler(payload: BatchRequest) -> Reconciler:
    """Validate the batch-level fields and build the reconciler.

    Shared by every endpoint: duplicate transaction ids across the merged
    sides are rejected with 400 and invalid excluded pairs with 422.
    """
    ids = [t.id for t in payload.left] + [t.id for t in payload.right]
    if len(set(ids)) != len(ids):
        raise HTTPException(status_code=400, detail="duplicate transaction id in batch")

    left = [Txn(id=t.id, ts=t.epoch_seconds(), amount=t.amount) for t in payload.left]
    right = [Txn(id=t.id, ts=t.epoch_seconds(), amount=t.amount) for t in payload.right]

    excluded: set[tuple[str, str]] = set()
    if payload.excluded_pairs:
        pairs_in = [
            ExcludedPair(left_id=p.left_id, right_id=p.right_id)
            for p in payload.excluded_pairs
        ]
        try:
            excluded = resolve_excluded_pairs(
                {t.id for t in left}, {t.id for t in right}, pairs_in
            )
        except ExcludedPairError as exc:
            raise RequestValidationError(
                [
                    {
                        "type": "excluded_pair.invalid",
                        "loc": ["body", "excluded_pairs", exc.index],
                        "msg": str(exc),
                        "input": payload.excluded_pairs[exc.index].model_dump(),
                    }
                ]
            )
    return Reconciler(left, right, excluded_pairs=excluded)


def solve(payload: ReconcileRequest) -> ReconcileResponse:
    """Validate the batch and compute the reconciliation result."""
    reconciler = build_reconciler(payload)

    locked_cands = []
    if payload.locked_groups:
        locked_in = [
            LockedGroup(left_ids=tuple(g.left_ids), right_ids=tuple(g.right_ids))
            for g in payload.locked_groups
        ]
        try:
            locked_cands = resolve_locked_groups(reconciler, locked_in)
        except LockedGroupError as exc:
            raise RequestValidationError(
                [
                    {
                        "type": "locked_group.invalid",
                        "loc": ["body", "locked_groups", exc.index],
                        "msg": str(exc),
                        "input": payload.locked_groups[exc.index].model_dump(),
                    }
                ]
            )

    groups = reconciler.solve(locked_cands)
    groups.sort(key=lambda c: c.encoding)

    matched_left = {i for c in groups for i in c.left_ids}
    matched_right = {i for c in groups for i in c.right_ids}

    return ReconcileResponse(
        groups=[
            GroupOut(left_ids=list(c.left_ids), right_ids=list(c.right_ids), amount=c.amount)
            for c in groups
        ],
        total_amount=sum(c.weight for c in groups),
        unmatched_left_ids=sorted(t.id for t in payload.left if t.id not in matched_left),
        unmatched_right_ids=sorted(t.id for t in payload.right if t.id not in matched_right),
    )


def _stored_response(body: str, replayed: bool) -> Response:
    # Return the stored bytes verbatim; ``Response`` instances bypass the
    # ``response_model`` serialization, so replays are byte-for-byte identical
    # to the first response.
    headers = {"Idempotency-Replayed": "true"} if replayed else None
    return Response(status_code=200, content=body, media_type="application/json", headers=headers)


@app.post(
    "/reconcile",
    response_model=ReconcileResponse,
    responses={
        409: {"description": "Idempotency-Key reused with a different request body."},
        200: {
            "headers": {
                "Idempotency-Replayed": {
                    "description": "'true' when a stored response was replayed for the key.",
                    "schema": {"type": "string", "enum": ["true"]},
                }
            }
        },
    },
)
def reconcile(
    payload: ReconcileRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> Any:
    # The endpoint behaves exactly as before when no key is supplied.
    if idempotency_key is None:
        return solve(payload)

    if not (IDEMPOTENCY_KEY_MIN <= len(idempotency_key) <= IDEMPOTENCY_KEY_MAX):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Idempotency-Key must be {IDEMPOTENCY_KEY_MIN}-"
                f"{IDEMPOTENCY_KEY_MAX} characters long"
            ),
        )

    fingerprint = canonical_fingerprint(payload)

    def producer() -> str:
        # Serialised with FastAPI's default JSON encoder so first-time and
        # replayed responses are byte-for-byte identical.
        return json.dumps(solve(payload).model_dump(), separators=(",", ":"), ensure_ascii=False)

    try:
        result = store.handle(idempotency_key, fingerprint, producer)
    except FingerprintConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return _stored_response(result.body, result.replayed)


@app.post("/reconcile/unmatched-summary", response_model=UnmatchedSummaryResponse)
def unmatched_summary(payload: UnmatchedSummaryRequest) -> UnmatchedSummaryResponse:
    """Aggregate unmatched flows by local day and amount sign.

    Runs the exact same validation and reconciliation as ``POST /reconcile``
    (same 400/422 behaviour), then attributes every unmatched transaction to
    its local calendar day in the requested IANA timezone -- DST transitions
    included -- and aggregates count and absolute amount per sign, plus
    whole-batch totals.
    """
    result = solve(payload)
    tz = ZoneInfo(payload.timezone)
    by_id = {t.id: t for t in payload.left + payload.right}

    per_day: dict[date, SignBreakdown] = {}
    total = SignBreakdown()
    for tid in result.unmatched_left_ids + result.unmatched_right_ids:
        txn = by_id[tid]
        day = datetime.fromtimestamp(txn.epoch_seconds(), tz=tz).date()
        bucket = per_day.setdefault(day, SignBreakdown())
        for acc in (bucket, total):
            sign = acc.positive if txn.amount > 0 else acc.negative
            sign.count += 1
            sign.amount += abs(txn.amount)

    return UnmatchedSummaryResponse(
        timezone=payload.timezone,
        days=[
            DaySummary(date=day, positive=b.positive, negative=b.negative)
            for day, b in sorted(per_day.items())
        ],
        total=total,
    )


@app.post("/reconcile/audit", response_model=AuditResponse)
def reconcile_audit(payload: AuditRequest) -> AuditResponse:
    """Audit an externally proposed complete reconciliation plan.

    The batch fields are validated exactly as in ``POST /reconcile`` (same
    400/422 behaviour).  Every proposed group is then checked against the
    usual rules -- references, duplicated ids, cross-group mutual exclusion,
    excluded pairs, shape, same sign, equal sums and the 48-hour span --
    collecting *all* failures keyed by the group's 0-based ``proposed_groups``
    index instead of aborting at the first one.

    A fully legal plan is compared against the canonical optimal solution of
    the batch: ``optimal`` tells whether it matches, ``difference``
    distinguishes an amount-suboptimal plan from one that reaches the optimal
    total but misses the lexicographically canonical grouping, and ``groups``
    carries the canonical solution for correction.  This endpoint is
    read-only and does not take part in ``Idempotency-Key`` replay.
    """
    reconciler = build_reconciler(payload)

    proposed = [
        LockedGroup(left_ids=tuple(g.left_ids), right_ids=tuple(g.right_ids))
        for g in payload.proposed_groups
    ]
    failures = audit_proposed_groups(reconciler, proposed)
    if failures:
        return AuditResponse(
            valid=False,
            errors=[AuditError(index=f.index, code=f.code, msg=str(f)) for f in failures],
            optimal=None,
            proposed_total=None,
            optimal_total=None,
            difference=None,
            groups=None,
        )

    canonical = reconciler.solve([])
    optimal_total = sum(c.weight for c in canonical)
    proposed_encodings = sorted(
        (tuple(sorted(g.left_ids)), tuple(sorted(g.right_ids))) for g in proposed
    )
    proposed_total = sum(reconciler.by_encoding[enc].weight for enc in proposed_encodings)
    optimal = proposed_encodings == [c.encoding for c in canonical]
    if optimal:
        difference = None
    elif proposed_total < optimal_total:
        difference = "suboptimal_amount"
    else:
        # A valid plan is a feasible packing, so its total can never exceed
        # the optimum; equal totals imply a non-canonical tie.
        difference = "non_canonical"
    return AuditResponse(
        valid=True,
        errors=[],
        optimal=optimal,
        proposed_total=proposed_total,
        optimal_total=optimal_total,
        difference=difference,
        groups=[
            GroupOut(left_ids=list(c.left_ids), right_ids=list(c.right_ids), amount=c.amount)
            for c in canonical
        ],
    )
