"""Compare release plotting inputs with the numerical fingerprint of the archived analyses."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

FILES = {
    "factor": "Fig2/input_conditions.csv",
    "excursion": "Fig2/steering_excursion.csv",
    "tyres": "Fig4/tyre_simulation.csv",
    "gain": "Fig5/local_gains.csv",
}
MODES = ["hold", "speed_only", "steer_only", "replay"]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value):
    try:
        return format(float(value), ".14g")
    except ValueError:
        return str(value)


def fingerprint(rows):
    normalized = sorted([[canonical(v) for v in row] for row in rows])
    blob = json.dumps(normalized, separators=(",", ":")).encode("ascii")
    return {"rows": len(rows), "sha256": hashlib.sha256(blob).hexdigest()}


def archive_fingerprints(publication, execution):
    rows = {key: [] for key in FILES}
    factor = load(publication / "factor/gpu_reproduction/aggregate.json")
    for item in factor["aggregate"]:
        if item["metric"] == "yaw_RMSE_radps" and item["steps"] == 30:
            panel = "a" if item["group"] == "sealed_id" else "b"
            rows["factor"].append([panel, item["group"], item["method"]] + [item["cells"][m] for m in MODES])
    for item in factor["paired"]:
        if item["metric"] == "yaw_RMSE_radps" and item["steps"] == 30 and item["group"] == "sealed_surface":
            for name, values in item["record_deltas"].items():
                rows["factor"].append(["c", item["group"], name] + [values[m] for m in MODES])
    for item in load(execution / "posthoc_vsd_20261116/regime_aggregate.json"):
        if (item["group"] == "sealed_surface" and item["mode"] == "hold"
                and item["method"] in ("data", "ode") and item["feature"] == "future_delta_change"):
            rows["excursion"].append([item["method"], *item["range"], item["records"],
                                      item["origins_per_seed"], item["seed_count"], item["rmse_vy_yaw"][1]])
    simulation = publication / "simulation/baseline_interface_repair"
    phases = load(simulation / "completion.json")["unique_control_seeds"]
    for item in load(simulation / "cell_summary.json"):
        for key, out in item["modes"].items():
            mode = {"speed": "speed_only", "steer": "steer_only", "both": "replay", "hold": "hold"}[key]
            rows["tyres"].append([item["plant"], item["front"], item["rear"], mode, out["data"][1],
                                  out["ode"][1], out["arx"][1], out["ode_vs_data_pct"][1], phases])
    stability = publication / "stability"
    inputs = [("results.json", "rows", ["gain", "gain0"]),
              ("bias_projection.json", "origins", ["gain_per_unit_stacked_norm"])]
    for filename, key, conditions in inputs:
        for item in load(stability / filename)[key]:
            for condition in conditions:
                for rank, value in enumerate(item[condition]):
                    rows["gain"].append([item["group"], item["arm"], condition, rank, item["record"], item["seed"], value])
    return {key: fingerprint(group) for key, group in rows.items()}


def release_fingerprints(figures):
    out = {}
    for key, name in FILES.items():
        with (figures / name).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            next(reader)
            out[key] = fingerprint(list(reader))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", type=Path, help="Optional original publication analysis directory")
    parser.add_argument("--execution", type=Path, help="Original KIT evaluation directory, with --publication")
    args = parser.parse_args()
    if bool(args.publication) != bool(args.execution):
        parser.error("--publication and --execution are used together")
    here = Path(__file__).resolve().parent
    registered = load(here / "figure_fingerprints.json")
    release = release_fingerprints(here.parent / "figures")
    if release != registered:
        raise SystemExit("Figure-input fingerprint mismatch")
    if args.publication and archive_fingerprints(args.publication, args.execution) != registered:
        raise SystemExit("Archived upstream analysis fingerprint mismatch")
    print(json.dumps(release, indent=2))


if __name__ == "__main__":
    main()
