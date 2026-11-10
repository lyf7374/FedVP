"""Apply the explicitly approved test-candidate failures without dropping attempts."""
import json
from pathlib import Path

AUTHORIZATION = "configs/heterogeneous_generation_failures_20261024.json"
FAILURE_REASON = "repetitive_generation_no_final_answer"
ROOT = Path(__file__).resolve().parents[1]


def candidate_failure_metadata(candidate, *, generator_name, dataset, split,
                               sample_idx, allow=False):
    marked = candidate.get("generation_failure") or candidate.get("selection_eligible") is False
    if not marked:
        if candidate.get("finish_reason") == "length":
            raise ValueError("Ordinary truncation is not an approved generation failure")
        return {}
    if not allow or split != "test":
        raise ValueError("Generation failures are accepted only by the approved test-selection path")
    if (candidate.get("generation_failure") is not True or
            candidate.get("selection_eligible") is not False or
            candidate.get("failure_reason") != FAILURE_REASON or
            candidate.get("failure_authorization") != AUTHORIZATION):
        raise ValueError("Candidate exclusion does not match the authorized failure policy")
    policy = json.loads((ROOT / AUTHORIZATION).read_text(encoding="utf-8"))
    identity = [sample_idx, candidate["candidate_id"]]
    if (policy.get("status") != "user_approved" or
            policy.get("generator_name") != generator_name or
            policy.get("dataset") != dataset or policy.get("split") != split or
            identity not in policy.get("candidate_ids", [])):
        raise ValueError("Candidate identity is not authorized for generation-failure exclusion")
    if (candidate.get("finish_reason") != "length" or
            candidate.get("extracted_answer") not in (None, "") or
            candidate.get("is_correct") not in (False, 0) or
            candidate.get("reward", 0) != 0):
        raise ValueError("Authorized loop record must retain its failed no-final-answer state")
    return dict(generation_failure=True, selection_eligible=False,
                failure_reason=FAILURE_REASON, failure_authorization=AUTHORIZATION)
