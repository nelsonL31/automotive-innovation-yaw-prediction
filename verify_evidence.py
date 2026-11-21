"""Verify the released file digests and the aggregated planned-steering results."""

import csv
import hashlib
import json
import math
from itertools import product
from pathlib import Path
from statistics import fmean


EVIDENCE = Path(__file__).resolve().parent / "evidence"
MODES = ("A", "B", "C")
MODELS = ("data", "ode")
MANEUVERS = ("slow", "varying")


def read_json(name):
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8-sig"))


def read_csv(name):
    with (EVIDENCE / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def match(value, expected, label):
    if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError(f"{label}: computed {value!r}; recorded {expected!r}")


def check_hashes():
    manifest = read_json("release_sha256.json")
    expected = {
        "README.md", "evidence.json", "gpu_audit_registration.json", "gpu_audit_result.json",
        "paired_trajectories.csv", "per_fit_trajectory.csv", "protocol.json", "summary.json",
        "reproduce/gpu_audit.py", "reproduce/planned.py",
    }
    require(manifest.keys() == expected, "The evidence file list differs from the released inventory.")
    for filename, expected_hash in manifest.items():
        observed_hash = hashlib.sha256((EVIDENCE / filename).read_bytes()).hexdigest()
        require(observed_hash == expected_hash, f"SHA-256 mismatch in {filename}")
    return manifest


def scenario_key(row):
    return float(row["mu"]), row["maneuver"], int(row["scene_seed"])


def paired_results(protocol):
    scenarios = set(product(protocol["mu"], MANEUVERS, protocol["seed_blocks"]))
    seeds = protocol["fit_seeds"]
    expected = set(product(scenarios, MODELS, seeds, MODES))
    fits = {}
    for row in read_csv("per_fit_trajectory.csv"):
        key = scenario_key(row), row["model"], int(row["fit_seed"]), row["mode"]
        require(key not in fits, f"Duplicate model/seed/mode observation: {key}")
        yaw, lateral = float(row["yaw_rmse_rad_s"]), float(row["lateral_speed_rmse_m_s"])
        require(all(math.isfinite(x) and x >= 0 for x in (yaw, lateral)), f"Invalid RMSE: {key}")
        fits[key] = yaw
    require(set(fits) == expected, "Missing or unexpected design cells in per-fit results.")
    pairs = {}
    for row in read_csv("paired_trajectories.csv"):
        scenario = scenario_key(row)
        require(scenario in scenarios and scenario not in pairs, f"Unexpected or duplicate trajectory: {scenario}")
        metrics = {}
        for mode in MODES:
            for model in MODELS:
                metrics[f"{model}_{mode}"] = fmean(fits[scenario, model, seed, mode] for seed in seeds)
            metrics[f"gap_{mode}"] = metrics[f"ode_{mode}"] - metrics[f"data_{mode}"]
        metrics["gap_B_minus_A"] = metrics["gap_B"] - metrics["gap_A"]
        for key, value in metrics.items():
            match(value, float(row[key]), f"{scenario} / {key}")
        for key in ("wheel_B_C_rms_rad", "issued_command_C_rms_rad"):
            value = float(row[key])
            require(math.isfinite(value) and value >= 0, f"Invalid steering RMS: {scenario} / {key}")
            metrics[key] = value
        pairs[scenario] = metrics
    require(set(pairs) == scenarios, "The paired table omits a scenario.")
    return fits, pairs


def check_group(rows, summary, name):
    rows = list(rows)
    calculated = {k: fmean(row[k] for row in rows) for k in rows[0]}
    for mode in MODES:
        calculated[f"ode_pct_{mode}"] = 100 * (calculated[f"ode_{mode}"] / calculated[f"data_{mode}"] - 1)
        calculated[f"ode_lower_n_{mode}"] = sum(row[f"gap_{mode}"] < 0 for row in rows)
    def sign(x):
        return (x > 0) - (x < 0)
    calculated["A_B_reversal_count"] = sum(sign(row["gap_A"]) != sign(row["gap_B"]) for row in rows)
    calculated["n"] = len(rows)
    require(calculated.keys() == summary.keys(), f"Unexpected summary fields in {name}")
    for field, value in calculated.items():
        match(value, summary[field], f"{name} / {field}")


def main():
    manifest = check_hashes()
    protocol = read_json("protocol.json")
    summary = read_json("summary.json")
    require(summary["protocol_sha"] == manifest["protocol.json"], "Analysis protocol digest differs.")
    fits, pairs = paired_results(protocol)
    require(len(fits) == summary["n_rows"] == 216, "Unanticipated number of fitted trajectory scores.")
    require(len(pairs) == summary["n_traj"] == 12, "Unanticipated number of simulated trajectories.")
    require(summary["n_origins"] == 14, "Unanticipated origin count in archived summary.")
    check_group(pairs.values(), summary["all"], "all trajectories")
    for name in MANEUVERS:
        check_group((values for (_, maneuver, _), values in pairs.items() if maneuver == name),
                    summary["by_maneuver"][name], name)
    print(f"Verified {len(manifest)} file hashes, {len(fits)} per-fit rows, {len(pairs)} paired trajectories and all three summaries.")
    print("ODE relative to data-only yaw-rate RMSE at 0.6 s:")
    for mode in MODES:
        print(f"  {mode}: {summary['all']['ode_pct_' + mode]:+.2f}%")


if __name__ == "__main__":
    main()

