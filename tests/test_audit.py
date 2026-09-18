"""API-level tests for the /reconcile/audit endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

T0 = "2026-09-17T00:00:00Z"


def txn(id: str, amount: int, time: str = T0) -> dict:
    return {"id": id, "time": time, "amount": amount}


def group(left_ids, right_ids) -> dict:
    return {"left_ids": list(left_ids), "right_ids": list(right_ids)}


def post_audit(left, right, proposed, **extra):
    return client.post(
        "/reconcile/audit",
        json={"left": left, "right": right, "proposed_groups": proposed, **extra},
    )


# --------------------------------------------------------------------- #
# valid plans
# --------------------------------------------------------------------- #
def test_optimal_plan():
    resp = post_audit(
        [txn("L1", 300)],
        [txn("R1", 100), txn("R2", 200)],
        [group(["L1"], ["R1", "R2"])],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is True
    assert body["errors"] == []
    assert body["optimal"] is True
    assert body["proposed_total"] == 300
    assert body["optimal_total"] == 300
    assert body["difference"] is None
    assert body["groups"] == [{"left_ids": ["L1"], "right_ids": ["R1", "R2"], "amount": 300}]


def test_empty_batch_and_empty_plan_is_optimal():
    resp = post_audit([], [], [])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body == {
        "valid": True,
        "errors": [],
        "optimal": True,
        "proposed_total": 0,
        "optimal_total": 0,
        "difference": None,
        "groups": [],
    }


def test_empty_plan_against_matchable_batch_is_suboptimal():
    resp = post_audit([txn("L1", 100)], [txn("R1", 100)], [])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is True
    assert body["optimal"] is False
    assert body["difference"] == "suboptimal_amount"
    assert body["proposed_total"] == 0
    assert body["optimal_total"] == 100
    assert body["groups"] == [{"left_ids": ["L1"], "right_ids": ["R1"], "amount": 100}]


def test_suboptimal_amount():
    # canonical optimum is (A+B,X)=150; the plan only reconciles (A,Y)=100
    resp = post_audit(
        [txn("A", 100), txn("B", 50)],
        [txn("X", 150), txn("Y", 100)],
        [group(["A"], ["Y"])],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is True
    assert body["optimal"] is False
    assert body["difference"] == "suboptimal_amount"
    assert body["proposed_total"] == 100
    assert body["optimal_total"] == 150
    assert body["groups"] == [{"left_ids": ["A", "B"], "right_ids": ["X"], "amount": 150}]


def test_partial_coverage_is_suboptimal():
    # the plan is legal but covers only one of the two canonical groups
    resp = post_audit(
        [txn("A", 100), txn("B", 100)],
        [txn("X", 100), txn("Y", 100)],
        [group(["A"], ["X"])],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is True
    assert body["optimal"] is False
    assert body["difference"] == "suboptimal_amount"
    assert body["proposed_total"] == 100
    assert body["optimal_total"] == 200


def test_same_total_but_non_canonical_grouping():
    # {(A,Y),(B,X)} reaches the optimal total 200, but the canonical
    # lexicographically smallest solution is {(A,X),(B,Y)}
    resp = post_audit(
        [txn("A", 100), txn("B", 100)],
        [txn("X", 100), txn("Y", 100)],
        [group(["A"], ["Y"]), group(["B"], ["X"])],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is True
    assert body["optimal"] is False
    assert body["difference"] == "non_canonical"
    assert body["proposed_total"] == 200
    assert body["optimal_total"] == 200
    assert body["groups"] == [
        {"left_ids": ["A"], "right_ids": ["X"], "amount": 100},
        {"left_ids": ["B"], "right_ids": ["Y"], "amount": 100},
    ]


def test_canonical_plan_with_unsorted_ids_is_optimal():
    # id order inside a proposed group does not matter
    resp = post_audit(
        [txn("A", 60), txn("B", 40)],
        [txn("X", 100)],
        [group(["B", "A"], ["X"])],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is True
    assert body["optimal"] is True
    assert body["proposed_total"] == 100
    assert body["groups"] == [{"left_ids": ["A", "B"], "right_ids": ["X"], "amount": 100}]


def test_negative_amounts_plan():
    resp = post_audit(
        [txn("L1", -150)],
        [txn("R1", -100), txn("R2", -50)],
        [group(["L1"], ["R1", "R2"])],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is True
    assert body["optimal"] is True
    assert body["proposed_total"] == 150
    assert body["optimal_total"] == 150
    assert body["groups"] == [{"left_ids": ["L1"], "right_ids": ["R1", "R2"], "amount": -150}]


def test_canonical_solution_respects_excluded_pairs():
    # forbidding (A,X) reroutes the canonical optimum to {(A,Y),(B,X)}
    resp = post_audit(
        [txn("A", 100), txn("B", 100)],
        [txn("X", 100), txn("Y", 100)],
        [group(["A"], ["Y"]), group(["B"], ["X"])],
        excluded_pairs=[{"left_id": "A", "right_id": "X"}],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is True
    assert body["optimal"] is True
    assert body["proposed_total"] == 200
    assert body["optimal_total"] == 200


def test_excluded_pair_can_make_plan_suboptimal():
    # natural optimum is (A+B,X)=150; forbidding (A,X) caps the optimum at 100
    resp = post_audit(
        [txn("A", 100), txn("B", 50)],
        [txn("X", 150), txn("Y", 100)],
        [group(["A"], ["Y"])],
        excluded_pairs=[{"left_id": "A", "right_id": "X"}],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is True
    assert body["optimal"] is True
    assert body["proposed_total"] == 100
    assert body["optimal_total"] == 100
    assert body["groups"] == [{"left_ids": ["A"], "right_ids": ["Y"], "amount": 100}]


# --------------------------------------------------------------------- #
# invalid plans: every check, all errors collected
# --------------------------------------------------------------------- #
def test_unknown_id_error():
    resp = post_audit([txn("A", 100)], [txn("X", 100)], [group(["A"], ["NOPE"])])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is False
    assert body["errors"] == [
        {
            "index": 0,
            "code": "unknown_id",
            "msg": "proposed_groups[0]: unknown transaction id: NOPE",
        }
    ]
    assert body["optimal"] is None
    assert body["proposed_total"] is None
    assert body["optimal_total"] is None
    assert body["difference"] is None
    assert body["groups"] is None


def test_duplicate_id_inside_group_error():
    resp = post_audit([txn("A", 100)], [txn("X", 100)], [group(["A", "A"], ["X"])])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is False
    assert body["errors"][0]["index"] == 0
    assert body["errors"][0]["code"] == "duplicate_id"


def test_reused_transaction_error_points_at_later_group():
    resp = post_audit(
        [txn("A", 100), txn("B", 100)],
        [txn("X", 100), txn("Y", 100)],
        [group(["A"], ["X"]), group(["A"], ["Y"]), group(["B"], ["Y"])],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is False
    assert [(e["index"], e["code"]) for e in body["errors"]] == [
        (1, "reused"),
        (2, "reused"),
    ]
    assert "A" in body["errors"][0]["msg"]
    assert "Y" in body["errors"][1]["msg"]


def test_excluded_pair_error():
    resp = post_audit(
        [txn("A", 100)],
        [txn("X", 100)],
        [group(["A"], ["X"])],
        excluded_pairs=[{"left_id": "A", "right_id": "X"}],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is False
    assert body["errors"][0]["code"] == "excluded_pair"
    assert "A" in body["errors"][0]["msg"] and "X" in body["errors"][0]["msg"]


def test_shape_error():
    # 2-2 is not one of the five legal shapes
    resp = post_audit(
        [txn("A", 60), txn("B", 40)],
        [txn("X", 50), txn("Y", 50)],
        [group(["A", "B"], ["X", "Y"])],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is False
    assert body["errors"][0]["code"] == "shape"
    assert "2-2" in body["errors"][0]["msg"]


def test_sign_error():
    resp = post_audit([txn("A", 100)], [txn("X", -100)], [group(["A"], ["X"])])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is False
    assert body["errors"][0]["code"] == "sign"


def test_sum_mismatch_error():
    resp = post_audit([txn("A", 100)], [txn("X", 101)], [group(["A"], ["X"])])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is False
    assert body["errors"][0]["code"] == "sum_mismatch"
    assert "100" in body["errors"][0]["msg"] and "101" in body["errors"][0]["msg"]


def test_span_error():
    resp = post_audit(
        [txn("A", 100, T0)],
        [txn("X", 100, "2026-09-19T00:00:01Z")],
        [group(["A"], ["X"])],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is False
    assert body["errors"][0]["code"] == "span"


def test_span_boundary_accepted():
    resp = post_audit(
        [txn("A", 100, T0)],
        [txn("X", 100, "2026-09-19T00:00:00Z")],
        [group(["A"], ["X"])],
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["valid"] is True
    assert resp.json()["optimal"] is True


def test_all_errors_collected_without_aborting():
    # group 0 unknown id, group 1 sum mismatch, group 2 bad shape: all three
    # are reported, keyed by their stable proposed_groups indices
    resp = post_audit(
        [txn("A", 100), txn("B", 100), txn("C", 1), txn("D", 1)],
        [txn("X", 101), txn("Y", 1), txn("Z", 1)],
        [
            group(["A"], ["NOPE"]),
            group(["B"], ["X"]),
            group(["C", "D"], ["Y", "Z"]),
        ],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is False
    assert [(e["index"], e["code"]) for e in body["errors"]] == [
        (0, "unknown_id"),
        (1, "sum_mismatch"),
        (2, "shape"),
    ]


def test_only_first_failing_check_reported_per_group():
    # the group is both mis-shapen (2-2) and sum-mismatched: only the shape
    # failure (first in check order) is reported for it
    resp = post_audit(
        [txn("A", 1), txn("B", 1)],
        [txn("X", 1), txn("Y", 2)],
        [group(["A", "B"], ["X", "Y"])],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is False
    assert [(e["index"], e["code"]) for e in body["errors"]] == [(0, "shape")]


def test_invalid_group_still_claims_its_transactions():
    # group 0 is sum-mismatched but still claims A; group 1 reusing A is flagged
    resp = post_audit(
        [txn("A", 100)],
        [txn("X", 101), txn("Y", 100)],
        [group(["A"], ["X"]), group(["A"], ["Y"])],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is False
    assert [(e["index"], e["code"]) for e in body["errors"]] == [
        (0, "sum_mismatch"),
        (1, "reused"),
    ]


def test_valid_groups_alongside_invalid_ones_do_not_leak_into_response():
    # group 0 is fine; only the broken ones appear in errors
    resp = post_audit(
        [txn("A", 100), txn("B", 100)],
        [txn("X", 100), txn("Y", 100)],
        [group(["A"], ["X"]), group(["B"], ["MISSING"])],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is False
    assert [(e["index"], e["code"]) for e in body["errors"]] == [(1, "unknown_id")]


# --------------------------------------------------------------------- #
# batch-level validation identical to /reconcile
# --------------------------------------------------------------------- #
def test_duplicate_batch_ids_return_400():
    resp = post_audit([txn("X", 1), txn("X", 2)], [], [])
    assert resp.status_code == 400
    cross = post_audit([txn("X", 1)], [txn("X", 1)], [])
    assert cross.status_code == 400


def test_invalid_excluded_pairs_return_422():
    resp = post_audit(
        [txn("A", 100)],
        [txn("X", 100)],
        [],
        excluded_pairs=[{"left_id": "NOPE", "right_id": "X"}],
    )
    assert resp.status_code == 422
    err = resp.json()["detail"][0]
    assert err["loc"] == ["body", "excluded_pairs", 0]
    assert err["type"] == "excluded_pair.invalid"


def test_proposed_groups_required():
    resp = client.post(
        "/reconcile/audit", json={"left": [txn("A", 100)], "right": [txn("X", 100)]}
    )
    assert resp.status_code == 422


def test_proposed_group_schema_errors_422():
    # missing side, empty side, too many ids and non-string ids are 422
    for bad in (
        {"right_ids": ["X"]},
        {"left_ids": [], "right_ids": ["X"]},
        {"left_ids": ["A", "B", "C", "D"], "right_ids": ["X"]},
        {"left_ids": ["A"], "right_ids": [1]},
    ):
        resp = post_audit([txn("A", 100)], [txn("X", 100)], [bad])
        assert resp.status_code == 422, bad
        assert "proposed_groups" in resp.json()["detail"][0]["loc"]


def test_more_than_12_per_side_rejected():
    resp = post_audit([txn(f"L{i}", 1) for i in range(13)], [], [])
    assert resp.status_code == 422


# --------------------------------------------------------------------- #
# original endpoints untouched
# --------------------------------------------------------------------- #
def test_reconcile_still_works():
    resp = client.post(
        "/reconcile", json={"left": [txn("L1", 100)], "right": [txn("R1", 100)]}
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["groups"] == [{"left_ids": ["L1"], "right_ids": ["R1"], "amount": 100}]
