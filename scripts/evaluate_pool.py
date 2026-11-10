"""CPU answer selection from complete saved candidates and verifier logits.

Uses the original stable scalar sigmoid and source candidate order, including
exact ties produced by sigmoid saturation. No model is loaded.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path


def sigmoid(value):
    if not math.isfinite(value):
        raise ValueError("Verifier logits must be finite")
    return 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))


def eligible(candidate):
    value = candidate.get("selection_eligible", True)
    if candidate.get("generation_failure", False) and value:
        raise ValueError("A generation failure cannot participate in selection")
    return value


def aligned(samples, member_logits, n, expected_questions):
    if len(samples) != expected_questions:
        raise ValueError(f"Expected {expected_questions} questions, got {len(samples)}")
    seen = set()
    chosen = []
    for sample in samples:
        sid = sample["sample_idx"]
        if sid in seen:
            raise ValueError(f"Duplicate question {sid}")
        seen.add(sid)
        candidates = sample["candidates"]
        ids = [c["candidate_id"] for c in candidates]
        if len(set(ids)) != len(ids) or len(ids) < n or ids != list(range(len(ids))):
            raise ValueError(f"Incomplete or unordered candidate IDs for question {sid}")
        chosen.append((sid, candidates[:n]))
    keys = {(sid, c["candidate_id"]) for sid, cs in chosen for c in cs}
    for name, scores in member_logits.items():
        missing = keys - scores.keys()
        if missing:
            raise ValueError(f"Missing {len(missing)} scores for {name}")
        for sid, candidates in chosen:
            for c in candidates:
                value = scores[(sid, c["candidate_id"])]
                if eligible(c):
                    if value is None or not math.isfinite(value):
                        raise ValueError(f"Missing/invalid score: {name}, {sid}, {c['candidate_id']}")
                elif value is not None:
                    raise ValueError("Excluded candidate must have null score")
    return chosen


def evaluate(samples, member_logits, *, n=8, expected_questions, mode="mean_probability", drop_member=None):
    if n not in (1, 2, 4, 8):
        raise ValueError("This release covers the paper's N=1,2,4,8 prefixes")
    members = dict(member_logits)
    if drop_member is not None:
        if drop_member not in members:
            raise ValueError(f"Unknown member {drop_member}")
        members.pop(drop_member)
    if mode not in ("sc", "oracle") and not members:
        raise ValueError("At least one verifier is required")
    rows = aligned(samples, members, n, expected_questions)
    predictions = []
    for sid, candidates in rows:
        choices = [c for c in candidates if eligible(c)]
        selected = None
        if choices:
            key = lambda c: (sid, c["candidate_id"])
            if mode == "sc":
                votes = Counter(c.get("extracted_answer") for c in choices if c.get("extracted_answer"))
                answer = max(votes, key=votes.get) if votes else None
                selected = next((c for c in choices if answer is not None and c.get("extracted_answer") == answer), None)
            elif mode == "oracle":
                selected = max(choices, key=lambda c: bool(c["is_correct"]))
            elif mode == "member_vote":
                votes = Counter(max(choices, key=lambda c: scores[key(c)])["candidate_id"] for scores in members.values())
                selected = max(choices, key=lambda c: votes[c["candidate_id"]] + 1e-6 * sum(sigmoid(scores[key(c)]) for scores in members.values()))
            else:
                def score(candidate):
                    values = [scores[key(candidate)] for scores in members.values()]
                    if mode == "mean_probability":
                        return sum(sigmoid(x) for x in values) / len(values)
                    if mode == "mean_logit":
                        return sum(values) / len(values)
                    if mode == "max_probability":
                        return max(sigmoid(x) for x in values)
                    raise ValueError(f"Unknown aggregation {mode}")
                selected = max(choices, key=score)
        predictions.append(dict(sample_idx=sid, candidate_id=None if selected is None else selected["candidate_id"],
                                is_correct=bool(selected and selected["is_correct"])))
    correct = sum(row["is_correct"] for row in predictions)
    return dict(status="complete", n=n, aggregation=mode, members=list(members),
                correct=correct, questions=expected_questions,
                accuracy_percent=100 * correct / expected_questions, predictions=predictions)


def read_scores(specifications):
    members = {}
    for spec in specifications:
        name, path = spec.split("=", 1)
        table = members.setdefault(name, {})
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = row["sample_idx"], row["candidate_id"]
            if key in table:
                raise ValueError(f"Duplicate score: {name}, {key}")
            table[key] = row["logit"]
    return members


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--scores", action="append", default=[], metavar="CLIENT=JSONL")
    parser.add_argument("--expected-questions", type=int, required=True)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--mode", choices=("mean_probability", "mean_logit", "max_probability", "member_vote", "sc", "oracle"), default="mean_probability")
    parser.add_argument("--drop-member")
    parser.add_argument("--teacher-labels", action="store_true", help="Export per-candidate mean teacher probabilities instead of test selections")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.expected_questions <= 0:
        parser.error("expected-questions must be positive")
    samples = json.loads(args.candidates.read_text(encoding="utf-8"))
    members = read_scores(args.scores)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.teacher_labels:
        if len(members) != 5 or args.drop_member or args.mode != "mean_probability" or args.n != 8:
            parser.error("Paper distillation uses all five teacher members and all eight candidates")
        rows = aligned(samples, members, args.n, args.expected_questions)
        labels = []
        for sid, candidates in rows:
            for c in candidates:
                if not eligible(c):
                    raise ValueError("Teacher training labels require complete eligible responses")
                key = (sid, c["candidate_id"])
                labels.append(dict(sample_idx=sid, candidate_id=c["candidate_id"],
                                   teacher_probability=sum(sigmoid(scores[key]) for scores in members.values()) / len(members)))
        args.output.write_text("".join(json.dumps(row) + "\n" for row in labels), encoding="utf-8")
        print(json.dumps(dict(status="complete", teacher_labels=len(labels), output=str(args.output))))
    else:
        result = evaluate(samples, members, n=args.n, expected_questions=args.expected_questions,
                          mode=args.mode, drop_member=args.drop_member)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k != "predictions"}))


if __name__ == "__main__":
    main()
