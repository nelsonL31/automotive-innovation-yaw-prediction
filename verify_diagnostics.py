"""Verify the released KIT sensitivity and racing summaries (Python 3.10+)."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from itertools import product
from pathlib import Path
from statistics import correlation, fmean, median


SEEDS = {"11", "22", "33"}
METRICS = {
    "v_(x) RMSE": "vx_rmse_m_per_s",
    "v_(x) MaxAE": "vx_maxae_m_per_s",
    "v_(y) RMSE": "vy_rmse_m_per_s",
    "v_(y) MaxAE": "vy_maxae_m_per_s",
    "r RMSE": "yaw_rate_rmse_rad_per_s",
    "r MaxAE": "yaw_rate_maxae_rad_per_s",
    "ADE": "ade_m",
    "FDE": "fde_m",
}
MODELS = ("ODE-PINN", "Reproduced PGNN")


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        reader.fieldnames = [" ".join(name.split()) for name in reader.fieldnames]
        return list(reader)


def require(condition, detail):
    if not condition:
        raise ValueError(str(detail))


def check_number(expected, observed, label, tolerance=1e-12):
    require(math.isclose(expected, float(observed), rel_tol=0, abs_tol=tolerance),
            f"{label}: computed {expected}, recorded {observed}")


def average_ranks(values):
    # Each drive/seed has nine origins. Ties receive the mean ordinal rank.
    ordered = sorted(values)
    return [ordered.index(value) + (ordered.count(value) + 1) / 2 for value in values]


def verify_sensitivity(data):
    fields = ("surface_group", "drive", "model", "seed")
    by_fit = defaultdict(list)
    for row in read_rows(data / "kit_sensitivity_by_origin.csv"):
        by_fit[tuple(row[f] for f in fields)].append(row)

    by_group = defaultdict(list)
    drive_seed = read_rows(data / "kit_sensitivity_by_drive_seed.csv")
    require(len(by_fit) == len(drive_seed) == 54, "Expected 54 drive/model/seed groups")
    for fit in drive_seed:
        key = tuple(fit[f] for f in fields)
        origins = by_fit.pop(key)
        require(len(origins) == len({r["origin_index"] for r in origins}) == 9, key)
        gains = [float(r["maximum_gain_standardised"]) for r in origins]
        errors = [float(r["endpoint_error_standardised"]) for r in origins]
        positives = [int(r["positive_spectral_abscissa_steps"]) for r in origins]
        require(all(int(r["sampled_steps"]) == 30 for r in origins), key)
        require(all(0 <= value <= 30 for value in positives), key)
        require(all(math.isfinite(value) and value >= 0 for value in gains + errors), key)
        rho = correlation(average_ranks(gains), average_ranks(errors))
        check_number(rho, fit["spearman_rho_nine_origins"], key)
        computed = (len(origins), sum(positives), 30 * len(origins), sum(x > 1 for x in gains))
        for value, field in zip(computed, ("n_origins", "positive_spectral_abscissa_steps",
                                          "sampled_steps", "origins_gain_gt_one")):
            check_number(value, fit[field], (key, field))
        by_group[(key[0], key[2])].append(fit)

    ngroups = {"same-surface": 5, "cross-surface": 4}
    require(set(by_group) == set(product(ngroups, ("Reference PGNN (GRU)", "Matched PGNN (MLP)"))),
            "Unexpected KIT group or model")
    for result in read_rows(data / "kit_sensitivity_summary.csv"):
        key = (result["surface_group"], result["model"])
        fits = by_group.pop(key)
        by_drive = defaultdict(dict)
        for fit in fits:
            by_drive[fit["drive"]][fit["seed"]] = float(fit["spearman_rho_nine_origins"])
        require(len(by_drive) == ngroups[key[0]] and all(set(v) == SEEDS for v in by_drive.values()), key)
        rho = median(median(seeds.values()) for seeds in by_drive.values())
        check_number(rho, result["drive_median_of_seed_median_rho"], key)
        for field in ("positive_spectral_abscissa_steps", "sampled_steps", "origins_gain_gt_one", "n_origins"):
            target = "total_origins_over_seeds" if field == "n_origins" else field
            check_number(sum(int(fit[field]) for fit in fits), result[target], (key, field))
        print(f"{key[0]} / {key[1]}: median Spearman = {rho:.3f}, "
              f"positive steps = {result['positive_spectral_abscissa_steps']}/{result['sampled_steps']}")
    require(not by_group, "Missing KIT summaries")


def read_seed_means(path, protocols, epochs):
    groups = defaultdict(dict)
    for row in read_rows(path):
        source = (row["training_record"], row["evaluation_records"])
        key = (*source, row["model"], row["epoch"])
        require(source in protocols and int(row["n_forecast_origins"]) == protocols[source], key)
        require(int(row["rollout_steps"]) == 15 and row["seed"] not in groups[key], key)
        groups[key][row["seed"]] = row
    expected = {(*source, model, epoch) for source, model, epoch in product(protocols, MODELS, epochs)}
    require(set(groups) == expected, "Unexpected racing model or protocol")
    means = {}
    for key, seeds in groups.items():
        require(set(seeds) == SEEDS, f"Unexpected seeds in {key}")
        for metric in METRICS.values():
            values = [float(r[metric]) for r in seeds.values()]
            require(all(math.isfinite(value) and value >= 0 for value in values), (key, metric))
            means[(*key, metric)] = fmean(values)
    return means


def percent(value, baseline):
    return 100 * (value - baseline) / baseline


def verify_racing(data, table):
    lvms_b = "LVMS-B historical re-evaluation"
    protocols = {("run4_2", "training prefix"): 9174, ("run2_1", lvms_b): 10017, ("run4_1", lvms_b): 10017}
    means = read_seed_means(data / "racecar_epoch_metrics_by_seed.csv", protocols, ("400", "2500"))
    for row in read_rows(data / "racecar_epoch_change_summary.csv"):
        group = (row["training_record"], row["evaluation_records"], row["model"])
        metric = row["metric"]
        early, late = (means.pop((*group, epoch, metric)) for epoch in ("400", "2500"))
        require(int(row["n_seeds"]) == 3, group)
        check_number(early, row["mean_epoch400"], (group, metric))
        check_number(late, row["mean_epoch2500"], (group, metric))
        check_number(percent(late, early), row["signed_change_percent"], (group, metric), 1e-10)
    require(not means, "Incomplete 400-to-2500 epoch summary")

    source = ("run4_2", "LVMS-A historical re-evaluation")
    means = read_seed_means(data / "racecar_lvmsa_final_by_seed.csv", {source: 13391}, ("2500",))
    for row in read_rows(table):
        metric = METRICS[row["Metric"]]
        ode, pgnn = (means.pop((*source, model, "2500", metric)) for model in MODELS)
        check_number(ode, row["ODE-PINN"], metric, 0.5e-6)
        check_number(pgnn, row["Reproduced PGNN"], metric, 0.5e-6)
        for comparator, denominator in (("published DDM", float(row["Published DDM"])), ("reproduced PGNN", pgnn)):
            shown = row["Change vs " + comparator].replace("−", "-").rstrip("%")
            check_number(percent(ode, denominator), shown, (metric, comparator), 0.005)
    require(not means, "Incomplete Table 3")
    print("Verified all 48 training-duration comparisons and all eight metrics and both percentage columns in Table 3.")


def verify_stability(root):
    physics = json.loads((root / "models/kit_contract.json").read_text(encoding="utf-8"))["physics"]
    m, inertia, lf, lr = (physics[k] for k in ("m", "Iz", "lf", "lr"))
    def entries(v, cf, cr):
        h = v**2 / ((v**2 + physics["speed_gate"]**2) * math.hypot(v, physics["vx_floor"]))
        k = lr * cr - lf * cf
        return (-(cf + cr) * h / m, k * h / m - v,
                k * h / inertia, -(lf**2 * cf + lr**2 * cr) * h / inertia)

    speeds = (2, 5, 10, 15, 20, 25)
    cases = set()
    for row in read_rows(root / "analysis/stability/fixed_stiffness_jacobians.csv"):
        v = float(row["speed_m_per_s"])
        cf, cr = (float(row[k]) for k in ("front_stiffness_N_per_rad", "rear_stiffness_N_per_rad"))
        case = (row["stiffness_setting"], v)
        require(v in speeds and case not in cases, case)
        cases.add(case)
        a, b, c, d = entries(v, cf, cr)
        for col, value in zip(("a11", "a12", "a21", "a22"), (a, b, c, d)):
            check_number(value, row[col], (case, col))
        determinant = a * d - b * c
        check_number(determinant, row["determinant_per_s_squared"], case, 1e-10)
        alpha = (a + d + math.sqrt(max(0.0, (a-d)**2 + 4*b*c))) / 2
        check_number(alpha, row["largest_real_eigenvalue_per_s"], case, 1e-10)
        require(row["locally_stable"] == str(alpha < 0).lower(), case)
    require(len(cases) == 48 and len({name for name, _ in cases}) == 8, "Incomplete stability grid")
    cr, cf = physics["bound_Cf_Cr"]
    low, high = 5.0, 10.0
    for _ in range(45):
        mid = (low + high) / 2
        a, b, c, d = entries(mid, cf, cr)
        if a * d - b * c > 0:
            low = mid
        else:
            high = mid
    check_number(mid, 7.65034, "critical speed (m/s)", 0.5e-5)
    print(f"Verified 48 fixed-stiffness cases and the reported critical speed ({mid:.5f} m/s).")


def verify_laps(root):
    rows = read_rows(root / "analysis/closed_loop/scale_racing_closed_loop_by_seed.csv")
    completions = {}
    for row in rows:
        key = (row["model"], row["seed"])
        require(key not in completions and row["completed"] in ("true", "false"), key)
        complete = row["completed"] == "true"
        require(complete == (row["end_of_attempt"] == "verified_lap"), key)
        require(complete == bool(row["lap_time_s"]), key)
        steps = int(row["executed_steps"])
        require(steps > 0 and 0 <= int(row["steps_over_20_ms"]) <= steps, key)
        require(0 <= int(row["boundary_excursions"]) <= int(row["outside_samples"]) <= steps, key)
        completions[key] = complete
    table = read_rows(root / "tables/table_4.csv")
    result, = [r for r in table if r["Measure"] == "Closed-loop laps completed / attempts"]
    models = set(result) - {"Measure"}
    require(set(completions) == set(product(models, SEEDS)), "Incomplete scale-racing design")
    for model in models:
        count = sum(completions[model, seed] for seed in SEEDS)
        require(result[model] == f"{count}/{len(SEEDS)}", model)
    print("Verified all 12 closed-loop records against the four Table 4 completion counts.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent, help="Repository root")
    root = parser.parse_args().root
    data = root / "analysis" / "diagnostics"
    manifest = json.loads((data / "sha256.json").read_text(encoding="utf-8-sig"))
    require(set(manifest) == {p.name for p in data.glob("*.csv")} | {"README.md"}, "Incomplete diagnostics manifest")
    for name, expected in manifest.items():
        require(hashlib.sha256((data / name).read_bytes()).hexdigest() == expected, f"SHA-256 mismatch: {name}")
    verify_sensitivity(data)
    verify_racing(data, root / "tables" / "table_3.csv")
    verify_laps(root)
    verify_stability(root)


if __name__ == "__main__":
    main()
