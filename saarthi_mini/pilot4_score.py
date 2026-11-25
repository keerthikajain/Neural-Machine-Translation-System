"""Score completed adjudicated Pilot 4 grades. No model or data generation."""
import argparse
from collections import Counter
from math import comb
from pathlib import Path
import random

from .data import read_rows, sha, write_json
from .pilot4_curate import metadata

SEED = 4042026
RESAMPLES = 10_000


def paired_stats(groups):
    differences = [value for group in groups for value in group]
    n = len(differences)
    if not n or any(not group for group in groups):
        raise ValueError("Empty paired stratum")
    counts = Counter(differences)
    wins, losses, ties = counts[1], counts[-1], counts[0]
    discordant = wins + losses
    p = min(1.0, 2 * sum(comb(discordant, i) for i in range(min(wins, losses) + 1))
            / 2 ** discordant)
    rng = random.Random(SEED)
    bootstrap = sorted(100 * sum(sum(rng.choices(group, k=len(group))) for group in groups) / n
                       for _ in range(RESAMPLES))
    return {"pairs": n, "wins": wins, "losses": losses, "ties": ties, "net_wins": wins - losses,
            "percentage_point_difference": 100 * (wins - losses) / n, "mcnemar_exact_p": p,
            "paired_bootstrap_95pp": [bootstrap[int((RESAMPLES - 1) * q)] for q in (.025, .975)]}


def evaluate(diagnostic, grades, control_chrf, treatment_chrf):
    if (len(diagnostic) != 60 or len({r["id"] for r in diagnostic}) != 60
            or Counter(r["category"] for r in diagnostic) != {"action": 30, "quantity": 30}):
        raise ValueError("Unexpected diagnostic layout")
    if [r["id"] for r in grades] != [r["id"] for r in diagnostic]:
        raise ValueError("Adjudicated results must match every frozen ID and order")
    results = {"action": [], "quantity": [], "negation": []}
    overall = {"action": [], "quantity": []}
    for item, pair in zip(diagnostic, grades, strict=True):
        category = item["category"]
        applicable = ("overall", category, "negation") if "negation" in item["tags"] else ("overall", category)
        for criterion in applicable:
            control, treatment = pair["control"][criterion], pair["treatment"][criterion]
            if type(control) is not bool or type(treatment) is not bool:
                raise ValueError(f"Incomplete adjudication on {criterion}")
            target = overall[category] if criterion == "overall" else results[criterion]
            target.append(treatment - control)
    if len(results["negation"]) < 10:
        raise ValueError("Insufficient scoped-negation cases")
    summary = {name: paired_stats([scores]) for name, scores in results.items()}
    summary["overall"] = paired_stats(list(overall.values()))
    criteria = {name: summary[name]["net_wins"] >= minimum for name, minimum in
                {"overall": 6, "action": 3, "quantity": 3, "negation": 0}.items()}
    criteria["validation_guardrail"] = treatment_chrf >= control_chrf - 0.5
    return {"judgments": summary, "validation_chrf++": {"control": control_chrf, "treatment": treatment_chrf},
            "criteria": criteria, "meets_outcome_thresholds": all(criteria.values())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("diagnostic", "adjudicated", "output"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    args = parser.parse_args()
    packet = metadata(args.adjudicated)
    if packet["diagnostic_sha256"] != sha(args.diagnostic) or packet["status"] != "adjudicated":
        raise ValueError("Adjudication status or diagnostic identity mismatch")
    result = evaluate(read_rows(args.diagnostic), packet["grades"],
                      packet["validation_chrf++"]["control"], packet["validation_chrf++"]["treatment"])
    result["diagnostic_sha256"] = packet["diagnostic_sha256"]
    result["adjudicated_file_sha256"] = sha(args.adjudicated)
    write_json(args.output, result)
