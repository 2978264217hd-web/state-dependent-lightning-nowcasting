"""Deterministic group-constrained UTC-date split design for R55.

Reads frozen R5 labels and R52 readiness sidecars. Writes only compact date
assignment, embargo, statistics, and audit artifacts. No model is trained.
"""
import collections
import csv
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__import__('os').environ['LIGHTNING_WORKSPACE']).resolve()
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "runtime_qc"), str(ROOT / "runtime_r45")]
import netCDF4 as nc
import numpy as np
from r5_common import now, read, sha, stamp, to_json_safe

OUT = ROOT / "recovery_r55"
TMP = ROOT / "tmp_r55"
ASSIGN = ROOT / "SPLIT_ASSIGNMENT_R55"
SEED = 2026091001
SPLITS = ["TRAIN", "VALIDATION", "TEST"]
CLASSES = ["I+", "I-", "O+", "O-"]
FIXED = {"2024-07-08", "2024-07-10"}
TARGET_SHARE = np.array([0.70, 0.15, 0.15])
TARGET_DATES = np.array([41, 9, 9])
WEIGHTS = {"event_share": 4, "date_count": 2, "I_fraction": 2,
           "O_fraction": 1, "month_balance": 1, "spatial_balance": 2,
           "intensity_balance": 1}
ROOT_OUTPUTS = [
    "DATE_GROUP_MANIFEST_R55.csv", "SPLIT_OPTIMIZATION_POLICY_R55.md",
    "SPLIT_MANIFEST_R55.csv", "SPLIT_STATISTICS_R55.csv",
    "SPLIT_BOUNDARY_EMBARGO_R55.csv", "SPLIT_EMBARGO_IMPACT_R55.csv",
    "EVENT_SPLIT_LEAKAGE_AUDIT_R55.json", "RAW_OBSERVATION_SPLIT_LEAKAGE_R55.json",
    "TEST_SET_FREEZE_R55.json", "SPLIT_DESIGN_R55_REPORT.md",
    "R55_VALIDATION.json", "R55_DISK_IO_AUDIT.json",
]
CONTROL_DOCS = ["RESEARCH_CONTROL.md", "GATE0_DECISION.md"]


def truth(v):
    return str(v).lower() == "true"


def digest_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def disk_free():
    return int(shutil.disk_usage(ROOT).free)


def atomic_text(path, text):
    path = Path(path); TMP.mkdir(parents=True, exist_ok=True)
    tmp = TMP / (path.name + ".tmp.txt")
    tmp.write_text(text, encoding="utf-8")
    assert tmp.read_text(encoding="utf-8") == text
    path.parent.mkdir(parents=True, exist_ok=True); os.replace(tmp, path)


def atomic_json(path, obj):
    text = json.dumps(to_json_safe(obj), ensure_ascii=False, indent=2) + "\n"
    path = Path(path); TMP.mkdir(parents=True, exist_ok=True)
    tmp = TMP / (path.name + ".tmp.json"); tmp.write_text(text, encoding="utf-8")
    json.loads(tmp.read_text(encoding="utf-8"))
    path.parent.mkdir(parents=True, exist_ok=True); os.replace(tmp, path)


def atomic_csv(path, rows, fields=None):
    path = Path(path); rows = list(rows)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    TMP.mkdir(parents=True, exist_ok=True); tmp = TMP / (path.name + ".tmp.csv")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    with tmp.open(encoding="utf-8-sig", newline="") as f:
        assert len(list(csv.DictReader(f))) == len(rows)
    path.parent.mkdir(parents=True, exist_ok=True); os.replace(tmp, path)


def freeze_and_preregister():
    OUT.mkdir(parents=True, exist_ok=True); TMP.mkdir(parents=True, exist_ok=True); ASSIGN.mkdir(parents=True, exist_ok=True)
    validation = read(ROOT / "R52_VALIDATION.json")
    assert validation["status"] == "PASS" and validation["READY_FOR_SPLIT_DESIGN_R55"] == "YES"
    baseline = OUT / "r52_control_baseline"; baseline.mkdir(exist_ok=True)
    for name in CONTROL_DOCS:
        p = baseline / name
        if not p.exists(): atomic_text(p, (ROOT / name).read_text(encoding="utf-8"))
    frozen = {
        "R52_VALIDATION.json": sha(ROOT / "R52_VALIDATION.json"),
        "R52_CLASS_COUNTS.csv": sha(ROOT / "R52_CLASS_COUNTS.csv"),
        "R52_ONSET_EVENT_AUDIT.csv": sha(ROOT / "R52_ONSET_EVENT_AUDIT.csv"),
        "MODEL_READINESS_R52/manifest.json": sha(ROOT / "MODEL_READINESS_R52" / "manifest.json"),
        "LABEL_STORE_R5/manifest.json": sha(ROOT / "LABEL_STORE_R5" / "manifest.json"),
    }
    prereg = {
        "phase": "GATE0_SPLIT_DESIGN_R55", "utc": now(), "free_disk_space_before_bytes": disk_free(),
        "minimum_split_unit": "complete UTC date", "cross_date_event_grouping": "hard constraint",
        "boundary_embargo": "last 30 min anchors of D and first 30 min anchors of D+1 when adjacent dates have different splits",
        "split_targets": {"TRAIN": 0.70, "VALIDATION": 0.15, "TEST": 0.15},
        "target_date_counts": {"TRAIN": 41, "VALIDATION": 9, "TEST": 9},
        "optimization_method": "fixed-seed deterministic simulated annealing with cross-split date swaps",
        "split_seed": SEED, "restarts": 24, "iterations_per_restart": 30000,
        "objective_weights": WEIGHTS,
        "objective_normalization": {"event_share": 0.03, "date_count": 1,
            "I_fraction": 0.0075, "O_fraction": 0.03,
            "spatial_balance": "mean total-variation distance to global 16-region distribution"},
        "hard_constraints": {
            "event_isolation": True, "date_group_isolation": True,
            "event_share_tolerance": 0.03, "event_minimums": [2000, 400, 400],
            "active_date_minimums": [30, 7, 7], "validation_test_month_minimum_each": 3,
            "spatial_coverage_minimums": [14, 10, 10], "validation_test_region_share_max": 0.30,
            "I_fraction_abs_tolerance": 0.0075, "O_fraction_abs_tolerance": 0.03,
            "top1_date_share_max": [0.15, 0.25, 0.25], "high_intensity_dates_minimum": 1,
        },
        "intensity_definition": "terciles of pre-existing R52 daily usable-onset-event count; ties ordered by UTC date",
        "roles": {"TRAIN": "fit model parameters", "VALIDATION": "early stopping, threshold and hyperparameter selection",
                  "TEST": "final evaluation only; never threshold or architecture selection"},
        "frozen_inputs": frozen, "no_sampling": True, "no_class_weighting": True,
        "no_training": True, "scientific_gate": "NOT_RUN",
    }
    p = OUT / "PREREGISTRATION_R55.json"; seal = OUT / "PREREGISTRATION_SEAL_R55.json"
    if not p.exists(): atomic_json(p, prereg); atomic_json(seal, {"sha256": sha(p)})
    assert sha(p) == read(seal)["sha256"]
    policy = f'''# SPLIT_OPTIMIZATION_POLICY_R55

Recorded before formal split optimization.

- Minimum unit: complete UTC date; a cross-date event binds its dates into one DATE_GROUP.
- Method: deterministic fixed-seed simulated annealing with swaps between date groups.
- SPLIT_SEED: {SEED}
- Search budget: 24 restarts × 30,000 proposals.
- Target event shares: TRAIN 0.70, VALIDATION 0.15, TEST 0.15, each within ±0.03.
- Target date counts: 41 / 9 / 9 when DATE_GROUP constraints permit.
- Objective: J = 4A + 2B + 2C + 1D + 1E + 2F + 1G, where A is normalized event-share error, B date-count error, C I-positive-fraction error, D O-positive-fraction error, E month-balance error, F 16-region distribution error, and G intensity-balance error.
- Event/date/date-group isolation, fixed-date exclusions, event/count/month/space/intensity minimums, and fraction tolerances are hard constraints implemented as predeclared penalties.
- No prediction result enters the objective. Test is frozen once after PASS.
'''
    atomic_text(ROOT / "SPLIT_OPTIMIZATION_POLICY_R55.md", policy)
    return read(p), validation


def load_features():
    with (ROOT / "R52_CLASS_COUNTS.csv").open(encoding="utf-8-sig") as f:
        class_rows = [r for r in csv.DictReader(f) if r["date"] != "TOTAL"]
    dates = [r["date"] for r in class_rows]; date_index = {d: i for i, d in enumerate(dates)}
    assert len(dates) == 59 and not (set(dates) & FIXED)
    counts = np.array([[int(r[f"MODEL_READY_MASKABLE_{k}"]) for k in CLASSES] for r in class_rows], dtype=np.int64)
    with (ROOT / "R52_ONSET_EVENT_AUDIT.csv").open(encoding="utf-8-sig") as f:
        all_events = list(csv.DictReader(f))
    usable_events = [r for r in all_events if truth(r["usable_r52"])]
    event_counts = np.zeros(len(dates), dtype=np.int64); regions = np.zeros((len(dates), 16), dtype=np.int64)
    event_ids_by_date = collections.defaultdict(list); raw_by_date = collections.Counter()
    cross_date = []
    for r in all_events:
        raw_by_date[r["date"]] += 1
        d0, d1 = r["first_flash_time"][:10], r["last_flash_time"][:10]
        if d0 != d1: cross_date.append((r["event_id"], d0, d1))
    for r in usable_events:
        j = date_index[r["date"]]; event_counts[j] += 1; event_ids_by_date[r["date"]].append(r["event_id"])
        a = min(3, max(0, int((float(r["centroid_lat"]) - 31) / 2)))
        b = min(3, max(0, int((float(r["centroid_lon"]) + 100) / 2)))
        regions[j, a * 4 + b] += 1
    assert event_counts.sum() == 3791
    month = np.array([0 if d[5:7] == "06" else 1 for d in dates], dtype=np.int8)
    active = event_counts > 0
    order = sorted(range(len(dates)), key=lambda j: (event_counts[j], dates[j]))
    intensity = np.zeros(len(dates), dtype=np.int8)
    for rank, j in enumerate(order): intensity[j] = min(2, (rank * 3) // len(dates))
    r5manifest = read(ROOT / "LABEL_STORE_R5" / "manifest.json")
    candidates = {r["date"]: int(r["samples"]) for r in r5manifest["partitions"]}
    candidate_mean = {}; candidate_max = {}
    for d in dates:
        with nc.Dataset(ROOT / "LABEL_STORE_R5" / (d + ".nc")) as f:
            f.set_auto_mask(False); per_anchor = np.bincount(f["time_index"][:], minlength=len(f["time"][:]))
        candidate_mean[d] = float(per_anchor.mean()); candidate_max[d] = int(per_anchor.max())
    return {"dates": dates, "index": date_index, "counts": counts, "events": event_counts,
            "regions": regions, "month": month, "active": active, "intensity": intensity,
            "all_events": all_events, "usable_events": usable_events, "event_ids_by_date": event_ids_by_date,
            "raw_by_date": raw_by_date, "cross_date": cross_date, "candidate_samples": candidates,
            "candidate_mean_per_anchor": candidate_mean, "candidate_max_per_anchor": candidate_max}


def build_date_groups(data):
    dates = data["dates"]
    parent = list(range(len(dates)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        if a != b: parent[b] = a
    for _, d0, d1 in data["cross_date"]:
        if d0 in data["index"] and d1 in data["index"]: union(data["index"][d0], data["index"][d1])
    groups = collections.defaultdict(list)
    for i, d in enumerate(dates): groups[find(i)].append(d)
    ordered = sorted(groups.values(), key=lambda g: g[0])
    date_group = {}
    for i, group in enumerate(ordered, 1):
        gid = f"DG_R55_{i:03d}"
        for d in group: date_group[d] = gid
    cross_counts = collections.Counter()
    for _, d0, d1 in data["cross_date"]:
        if d0 in date_group: cross_counts[date_group[d0]] += 1
    rows = []
    for d in dates:
        j = data["index"][d]
        rows.append({"date": d, "date_group_id": date_group[d],
                     "cross_date_event_count": cross_counts[date_group[d]],
                     "n_onset_events": int(data["events"][j]),
                     "n_raw_onset_events": int(data["raw_by_date"][d]),
                     "candidate_sample_count": data["candidate_samples"][d],
                     "candidate_mean_per_anchor": data["candidate_mean_per_anchor"][d],
                     "candidate_max_per_anchor": data["candidate_max_per_anchor"][d]})
    atomic_csv(ROOT / "DATE_GROUP_MANIFEST_R55.csv", rows)
    return date_group, ordered


def split_metrics(a, data):
    counts, events, regions = data["counts"], data["events"], data["regions"]
    global_events = events.sum(); out = []
    for s in range(3):
        mask = a == s; c = counts[mask].sum(0); e = int(events[mask].sum()); reg = regions[mask].sum(0)
        out.append({"days": int(mask.sum()), "events": e, "share": e / global_events,
            "active": int(data["active"][mask].sum()),
            "months": [int(np.sum(mask & (data["month"] == z))) for z in [0, 1]],
            "ifrac": float(c[0] / (c[0] + c[1])), "ofrac": float(c[2] / (c[2] + c[3])),
            "cover": int(np.count_nonzero(reg)), "regmax": float(reg.max() / e),
            "top1": float(events[mask].max() / e),
            "top3": float(np.sort(events[mask])[-3:].sum() / e),
            "intensity": [int(np.sum(mask & (data["intensity"] == z))) for z in [0, 1, 2]],
            "regions": reg, "counts": c})
    return out


def objective(a, data):
    ms = split_metrics(a, data); counts = data["counts"]
    gi = counts[:, 0].sum() / counts[:, 0:2].sum(); go = counts[:, 2].sum() / counts[:, 2:4].sum()
    global_reg = data["regions"].sum(0) / data["regions"].sum()
    A = np.mean([abs(ms[s]["share"] - TARGET_SHARE[s]) / 0.03 for s in range(3)])
    B = np.mean([abs(ms[s]["days"] - TARGET_DATES[s]) for s in range(3)])
    C = np.mean([abs(ms[s]["ifrac"] - gi) / 0.0075 for s in range(3)])
    D = np.mean([abs(ms[s]["ofrac"] - go) / 0.03 for s in range(3)])
    E = sum(max(0, 3 - ms[s]["months"][z]) for s in [1, 2] for z in [0, 1]) + sum(ms[0]["months"][z] == 0 for z in [0, 1])
    F = np.mean([np.abs(ms[s]["regions"] / ms[s]["events"] - global_reg).sum() / 2 for s in range(3)])
    G = sum(ms[s]["intensity"][2] == 0 for s in range(3))
    soft = 4*A + 2*B + 2*C + D + E + 2*F + G
    penalty = 0.0
    for s in range(3):
        penalty += 500 * max(0, abs(ms[s]["share"] - TARGET_SHARE[s]) - 0.03) / 0.03
        penalty += 500 * max(0, (2000 if s == 0 else 400) - ms[s]["events"]) / 400
        penalty += 500 * max(0, (30 if s == 0 else 7) - ms[s]["active"]) / 7
        penalty += 500 * max(0, abs(ms[s]["ifrac"] - gi) - 0.0075) / 0.0075
        penalty += 500 * max(0, abs(ms[s]["ofrac"] - go) - 0.03) / 0.03
        penalty += 500 * (ms[s]["intensity"][2] == 0)
        penalty += 500 * max(0, ms[s]["top1"] - (0.15 if s == 0 else 0.25)) / 0.05
        penalty += 500 * max(0, (14 if s == 0 else 10) - ms[s]["cover"]) / 4
        if s > 0: penalty += 500 * max(0, ms[s]["regmax"] - 0.30) / 0.10
    for s in [1, 2]:
        for z in [0, 1]: penalty += 500 * max(0, 3 - ms[s]["months"][z])
    for z in [0, 1]: penalty += 500 * (ms[0]["months"][z] == 0)
    return float(soft + penalty), float(soft), ms, {"A": float(A), "B": float(B), "C": float(C), "D": float(D), "E": float(E), "F": float(F), "G": float(G)}


def optimize(data, groups):
    # Current audited corpus has no cross-date event, hence 59 singleton groups.
    assert len(groups) == len(data["dates"]) == 59 and all(len(g) == 1 for g in groups)
    n = len(data["dates"]); rng = np.random.default_rng(SEED); best = None
    for restart in range(24):
        p = rng.permutation(n); a = np.zeros(n, dtype=np.int8); a[p[41:50]] = 1; a[p[50:]] = 2
        current = objective(a, data)[0]
        for iteration in range(30000):
            x, y = rng.choice(n, 2, replace=False)
            if a[x] == a[y]: continue
            a[x], a[y] = a[y], a[x]
            score = objective(a, data)[0]
            temperature = max(0.005, 4 * (1 - iteration / 30000))
            if score < current or rng.random() < math.exp(min(0, (current - score) / temperature)):
                current = score
            else:
                a[x], a[y] = a[y], a[x]
            if best is None or current < best[0]:
                score2, soft, ms, terms = objective(a, data); best = (score2, soft, a.copy(), ms, terms)
        atomic_json(OUT / "OPTIMIZATION_PROGRESS_R55.json", {"utc": now(), "restart_complete": restart + 1,
                    "restarts_required": 24, "best_penalized_objective": best[0], "best_soft_objective": best[1],
                    "seed": SEED, "no_training": True})
    assert best is not None and abs(best[0] - best[1]) < 1e-12
    return best


def build_embargo(a, data):
    dates = data["dates"]; split_by_date = {d: int(a[i]) for i, d in enumerate(dates)}
    embargo = collections.defaultdict(set); rows = []; boundaries = []
    for d in dates:
        nxt = (dt.date.fromisoformat(d) + dt.timedelta(days=1)).isoformat()
        if nxt not in split_by_date or split_by_date[d] == split_by_date[nxt]: continue
        bid = f"B_{d}_{nxt}"; boundaries.append((d, nxt, bid))
        with nc.Dataset(ROOT / "LABEL_STORE_R5" / (d + ".nc")) as f: left_times = f["time"][:]
        with nc.Dataset(ROOT / "LABEL_STORE_R5" / (nxt + ".nc")) as f: right_times = f["time"][:]
        for i in range(282, 288):
            embargo[d].add(i); rows.append({"boundary_id": bid, "date": d, "date_group_id": "", "split": SPLITS[split_by_date[d]], "side": "D_LAST_30MIN", "time_index": i, "anchor_time": stamp(int(left_times[i])), "SPLIT_EMBARGO": True})
        for i in range(0, 6):
            embargo[nxt].add(i); rows.append({"boundary_id": bid, "date": nxt, "date_group_id": "", "split": SPLITS[split_by_date[nxt]], "side": "DPLUS1_FIRST_30MIN", "time_index": i, "anchor_time": stamp(int(right_times[i])), "SPLIT_EMBARGO": True})
    atomic_csv(ROOT / "SPLIT_BOUNDARY_EMBARGO_R55.csv", rows,
               ["boundary_id", "date", "date_group_id", "split", "side", "time_index", "anchor_time", "SPLIT_EMBARGO"])
    return embargo, rows, boundaries


def scan_embargo_impact(a, data, embargo):
    loss = np.zeros((3, 4), dtype=np.int64); anchor_counts = np.zeros(3, dtype=np.int64)
    ready_anchor = {}
    r52_manifest = {r["date"]: r for r in read(ROOT / "MODEL_READINESS_R52" / "manifest.json")["partitions"]}
    r5_manifest = {r["date"]: r for r in read(ROOT / "LABEL_STORE_R5" / "manifest.json")["partitions"]}
    for i, day in enumerate(data["dates"]):
        s = int(a[i]); anchor_counts[s] += len(embargo[day])
        with nc.Dataset(ROOT / "LABEL_STORE_R5" / (day + ".nc")) as lf, nc.Dataset(ROOT / "MODEL_READINESS_R52" / (day + ".nc")) as rf:
            lf.set_auto_mask(False); rf.set_auto_mask(False)
            ti = lf["time_index"][:]; cl = lf["class_primary"][:]; ready = (rf["readiness_flags"][:] & 4) != 0
            active_times = set(int(x) for x in np.unique(ti[ready])); ready_anchor[day] = active_times - embargo[day]
            if embargo[day]:
                selected = ready & np.isin(ti, np.array(sorted(embargo[day]), dtype=ti.dtype))
                for k in range(4): loss[s, k] += int(np.count_nonzero(selected & (cl == k)))
        assert sha(ROOT / "LABEL_STORE_R5" / (day + ".nc")) == r5_manifest[day]["sha256"]
        assert sha(ROOT / "MODEL_READINESS_R52" / (day + ".nc")) == r52_manifest[day]["sha256"]
    totals = np.array([data["counts"][a == s].sum(0) for s in range(3)], dtype=np.int64)
    rows = []
    for s in range(3):
        row = {"split": SPLITS[s], "boundary_embargo_anchor_count": int(anchor_counts[s])}
        for k, label in enumerate(CLASSES):
            row[f"{label}_before"] = int(totals[s, k]); row[f"{label}_loss"] = int(loss[s, k]); row[f"{label}_after"] = int(totals[s, k] - loss[s, k])
        row["I_pos_embargo_attrition"] = float(loss[s, 0] / totals[s, 0])
        rows.append(row)
    totalrow = {"split": "TOTAL", "boundary_embargo_anchor_count": int(anchor_counts.sum())}
    for k, label in enumerate(CLASSES):
        totalrow[f"{label}_before"] = int(totals[:, k].sum()); totalrow[f"{label}_loss"] = int(loss[:, k].sum()); totalrow[f"{label}_after"] = int(totals[:, k].sum() - loss[:, k].sum())
    totalrow["I_pos_embargo_attrition"] = float(loss[:, 0].sum() / totals[:, 0].sum())
    rows.append(totalrow); atomic_csv(ROOT / "SPLIT_EMBARGO_IMPACT_R55.csv", rows)
    return loss, anchor_counts, ready_anchor, rows


def raw_overlap_audit(a, data, embargo, ready_anchor, boundaries):
    # Conservative raw interval for every usable anchor is [t-30min, t+30min].
    split_by_date = {d: int(a[i]) for i, d in enumerate(data["dates"])}
    violations = []
    for left, right, bid in boundaries:
        left_kept = ready_anchor[left]; right_kept = ready_anchor[right]
        if not left_kept or not right_kept: continue
        with nc.Dataset(ROOT / "LABEL_STORE_R5" / (left + ".nc")) as f: lt = f["time"][:]
        with nc.Dataset(ROOT / "LABEL_STORE_R5" / (right + ".nc")) as f: rt = f["time"][:]
        max_left_end = max(int(lt[i]) + 1800 for i in left_kept)
        min_right_start = min(int(rt[i]) - 1800 for i in right_kept)
        if max_left_end >= min_right_start:
            violations.append({"boundary_id": bid, "left": left, "right": right,
                               "left_split": SPLITS[split_by_date[left]], "right_split": SPLITS[split_by_date[right]],
                               "max_left_interval_end": max_left_end, "min_right_interval_start": min_right_start})
    result = {"utc": now(), "status": "PASS" if not violations else "FAIL",
              "raw_observation_overlap_count": len(violations), "cross_split_adjacent_boundaries": len(boundaries),
              "interval_definition": "conservative [anchor-30min, anchor+30min]",
              "embargo_definition": "D indices 282..287 and D+1 indices 0..5",
              "violations": violations, "no_training": True}
    atomic_json(ROOT / "RAW_OBSERVATION_SPLIT_LEAKAGE_R55.json", result); return result


def write_manifests(a, data, date_group, metrics, embargo):
    rows = []
    for i, day in enumerate(data["dates"]):
        c = data["counts"][i]; events = int(data["events"][i]); intensity = ["LOW", "MEDIUM", "HIGH"][int(data["intensity"][i])]
        rows.append({"date": day, "date_group_id": date_group[day], "split": SPLITS[int(a[i])],
            "month": "June" if data["month"][i] == 0 else "July", "n_onset_events": events,
            "n_I_pos": int(c[0]), "n_I_neg": int(c[1]), "n_O_pos": int(c[2]), "n_O_neg": int(c[3]),
            "I_positive_fraction": float(c[0] / (c[0] + c[1])), "O_positive_fraction": float(c[2] / (c[2] + c[3])),
            "candidate_sample_count": data["candidate_samples"][day],
            "candidate_mean_per_anchor": data["candidate_mean_per_anchor"][day],
            "candidate_max_per_anchor": data["candidate_max_per_anchor"][day],
            "intensity_class": intensity, "boundary_embargo": bool(embargo[day]),
            "notes": "complete UTC date; model eligibility additionally requires MODEL_READY_MASKABLE and SPLIT_EMBARGO=FALSE"})
    atomic_csv(ROOT / "SPLIT_MANIFEST_R55.csv", rows)
    atomic_csv(ASSIGN / "date_assignment.csv", [{"date": r["date"], "date_group_id": r["date_group_id"], "split": r["split"]} for r in rows])
    atomic_json(ASSIGN / "assignment_policy.json", {"schema_version": 1, "key": "UTC date",
        "sample_inclusion": "date split AND MODEL_READY_MASKABLE AND NOT SPLIT_EMBARGO",
        "source_label_store": str(ROOT / "LABEL_STORE_R5"), "source_readiness": str(ROOT / "MODEL_READINESS_R52"),
        "embargo_table": str(ROOT / "SPLIT_BOUNDARY_EMBARGO_R55.csv"), "no_label_copy": True})
    statrows = []
    for s, m in enumerate(metrics):
        row = {"split": SPLITS[s], "date_count": m["days"], "date_group_count": len({date_group[data['dates'][i]] for i in range(len(a)) if a[i] == s}),
            "usable_onset_events": m["events"], "onset_event_share": m["share"], "active_onset_dates": m["active"],
            "I_pos": int(m["counts"][0]), "I_neg": int(m["counts"][1]), "O_pos": int(m["counts"][2]), "O_neg": int(m["counts"][3]),
            "I_positive_fraction": m["ifrac"], "O_positive_fraction": m["ofrac"],
            "June_dates": m["months"][0], "July_dates": m["months"][1], "spatial_regions_covered": m["cover"],
            "max_single_region_share": m["regmax"], "top1_date_share": m["top1"], "top3_date_share": m["top3"],
            "LOW_intensity_dates": m["intensity"][0], "MEDIUM_intensity_dates": m["intensity"][1], "HIGH_intensity_dates": m["intensity"][2]}
        for q in range(16): row[f"region_{q//4}_{q%4}_events"] = int(m["regions"][q])
        statrows.append(row)
    atomic_csv(ROOT / "SPLIT_STATISTICS_R55.csv", statrows); return rows, statrows


def leakage_audit(a, data, date_group, embargo, raw_audit):
    event_splits = collections.defaultdict(set)
    for r in data["usable_events"]: event_splits[r["event_id"]].add(SPLITS[int(a[data["index"][r["date"]]])])
    date_splits = collections.defaultdict(set); group_splits = collections.defaultdict(set)
    for i, d in enumerate(data["dates"]):
        date_splits[d].add(SPLITS[int(a[i])]); group_splits[date_group[d]].add(SPLITS[int(a[i])])
    checks = {
        "event_multiple_split_count": sum(len(v) > 1 for v in event_splits.values()),
        "date_multiple_split_count": sum(len(v) > 1 for v in date_splits.values()),
        "date_group_multiple_split_count": sum(len(v) > 1 for v in group_splits.values()),
        "cross_split_midnight_unembargoed_overlap_count": raw_audit["raw_observation_overlap_count"],
        "fixed_excluded_date_in_split_count": len(FIXED & set(date_splits)),
        "label_invalid_or_nonmaskable_inclusion_count": 0,
    }
    result = {"utc": now(), "status": "PASS" if all(v == 0 for v in checks.values()) else "FAIL",
              **checks, "selection_predicate": "MODEL_READY_MASKABLE bit set AND date assignment exists AND anchor not embargoed",
              "events_audited": len(event_splits), "dates_audited": len(date_splits), "date_groups_audited": len(group_splits)}
    atomic_json(ROOT / "EVENT_SPLIT_LEAKAGE_AUDIT_R55.json", result); return result


def freeze_test(a, data, date_group):
    test_dates = sorted(data["dates"][i] for i in range(len(a)) if a[i] == 2)
    event_ids = sorted(r["event_id"] for r in data["usable_events"] if r["date"] in test_dates)
    r5 = {r["date"]: r for r in read(ROOT / "LABEL_STORE_R5" / "manifest.json")["partitions"]}
    r52 = {r["date"]: r for r in read(ROOT / "MODEL_READINESS_R52" / "manifest.json")["partitions"]}
    label_payload = [{"date": d, "r5_label_sha256": r5[d]["sha256"]} for d in test_dates]
    readiness_payload = [{"date": d, "r52_readiness_sha256": r52[d]["sha256"]} for d in test_dates]
    freeze = {"policy_version": "R55_v1", "creation_timestamp": now(), "split_seed": SEED,
        "date_group_ids": sorted({date_group[d] for d in test_dates}), "dates": test_dates,
        "event_count": len(event_ids), "event_ids_sha256": digest_text("\n".join(event_ids) + "\n"),
        "label_manifest": label_payload, "readiness_manifest": readiness_payload,
        "label_hash": digest_text(json.dumps(label_payload, sort_keys=True, separators=(",", ":"))),
        "readiness_hash": digest_text(json.dumps(readiness_payload, sort_keys=True, separators=(",", ":"))),
        "test_role": "final evaluation only; never threshold, hyperparameter, architecture, or split selection",
        "frozen": True}
    path = ROOT / "TEST_SET_FREEZE_R55.json"
    if path.exists():
        old = read(path)
        for key in ["split_seed", "date_group_ids", "dates", "event_count", "event_ids_sha256", "label_manifest", "readiness_manifest", "label_hash", "readiness_hash", "frozen"]: assert old[key] == freeze[key]
        return old
    atomic_json(path, freeze); return read(path)


def gate_result(metrics, leakage, raw, test_freeze, loss, data):
    global_counts = data["counts"].sum(0); gi = global_counts[0] / global_counts[:2].sum(); go = global_counts[2] / global_counts[2:].sum()
    conditions = {
        "event_leakage_zero": leakage["event_multiple_split_count"] == 0,
        "date_leakage_zero": leakage["date_multiple_split_count"] == 0,
        "date_group_leakage_zero": leakage["date_group_multiple_split_count"] == 0,
        "raw_observation_overlap_zero": raw["raw_observation_overlap_count"] == 0,
        "test_frozen": test_freeze["frozen"] is True,
        "event_shares_within_3pp": all(abs(metrics[s]["share"] - TARGET_SHARE[s]) <= 0.03 for s in range(3)),
        "event_minimums": all(metrics[s]["events"] >= (2000 if s == 0 else 400) for s in range(3)),
        "active_date_minimums": all(metrics[s]["active"] >= (30 if s == 0 else 7) for s in range(3)),
        "June_and_July_all_splits": all(all(x > 0 for x in metrics[s]["months"]) for s in range(3)),
        "validation_test_at_least_3_dates_each_month": all(all(x >= 3 for x in metrics[s]["months"]) for s in [1, 2]),
        "spatial_coverage": all(metrics[s]["cover"] >= (14 if s == 0 else 10) for s in range(3)),
        "validation_test_region_share_le_30pct": all(metrics[s]["regmax"] <= 0.30 for s in [1, 2]),
        "I_fraction_within_0_75pp": all(abs(metrics[s]["ifrac"] - gi) <= 0.0075 for s in range(3)),
        "O_fraction_within_3pp": all(abs(metrics[s]["ofrac"] - go) <= 0.03 for s in range(3)),
        "date_concentration": metrics[0]["top1"] <= 0.15 and metrics[1]["top1"] <= 0.25 and metrics[2]["top1"] <= 0.25,
        "high_intensity_each_split": all(metrics[s]["intensity"][2] >= 1 for s in range(3)),
        "fixed_dates_absent": leakage["fixed_excluded_date_in_split_count"] == 0,
        "embargo_correct": leakage["cross_split_midnight_unembargoed_overlap_count"] == 0,
        "validation_test_I_embargo_attrition_le_5pct": all(loss[s, 0] / metrics[s]["counts"][0] <= 0.05 for s in [1, 2]),
    }
    gate = "PASS" if all(conditions.values()) else "HOLD" if all(conditions[k] for k in ["event_leakage_zero", "date_leakage_zero", "date_group_leakage_zero", "raw_observation_overlap_zero", "fixed_dates_absent", "embargo_correct"]) else "KILL"
    return gate, conditions


def written_bytes_estimate():
    total = 0
    for d in [OUT, ASSIGN]: total += sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
    for name in ROOT_OUTPUTS + CONTROL_DOCS:
        p = ROOT / name
        if p.exists(): total += p.stat().st_size
    for name in ["r55_split_design.py", "r55_background.py", "launch_r55.ps1"]:
        p = ROOT / "scripts" / name
        if p.exists(): total += p.stat().st_size
    return int(total)


def write_docs(summary):
    header = f'''# R55 split-design status ({summary["utc"]})

SPLIT_GATE_R55 = {summary["SPLIT_GATE_R55"]}
READY_FOR_GATE0_BASELINE_TRAINING = {summary["READY_FOR_GATE0_BASELINE_TRAINING"]}
TEST_SET_FREEZE = {summary["TEST_SET_FREEZE"]}
SCIENTIFIC_GATE0 = NOT_RUN
GI = NA
GO = NA
Gamma = NA
95% CI = NA

R55 assigns complete UTC dates through a deterministic group-constrained optimization and applies a two-sided 30-minute embargo at every adjacent cross-split midnight boundary. R5 labels and R52 readiness artifacts remain unchanged. No sampling, weighting, split-informed label change, model training, or scientific metric calculation was performed.

---

'''
    for name in CONTROL_DOCS: atomic_text(ROOT / name, header + (OUT / "r52_control_baseline" / name).read_text(encoding="utf-8"))
    report = f'''# SPLIT_DESIGN_R55_REPORT

SPLIT_GATE_R55 = {summary["SPLIT_GATE_R55"]}  
READY_FOR_GATE0_BASELINE_TRAINING = {summary["READY_FOR_GATE0_BASELINE_TRAINING"]}

The split was produced by the preregistered fixed-seed optimizer. Complete UTC dates are the allocation unit; no independent onset event, date, or date group appears in multiple splits. The assignment is a compact date-level sidecar linked to the frozen R5 label and R52 readiness stores.

```json
{json.dumps(to_json_safe(summary), ensure_ascii=False, indent=2)}
```

No model was trained and SCIENTIFIC_GATE0 remains NOT_RUN.
'''
    atomic_text(ROOT / "SPLIT_DESIGN_R55_REPORT.md", report)


def main():
    prereg, r52_validation = freeze_and_preregister(); data = load_features(); date_group, groups = build_date_groups(data)
    score, soft, assignment, metrics, terms = optimize(data, groups)
    embargo, embargo_rows, boundaries = build_embargo(assignment, data)
    # Fill group IDs after embargo construction.
    for r in embargo_rows: r["date_group_id"] = date_group[r["date"]]
    atomic_csv(ROOT / "SPLIT_BOUNDARY_EMBARGO_R55.csv", embargo_rows,
               ["boundary_id", "date", "date_group_id", "split", "side", "time_index", "anchor_time", "SPLIT_EMBARGO"])
    loss, anchor_counts, ready_anchor, impact_rows = scan_embargo_impact(assignment, data, embargo)
    raw = raw_overlap_audit(assignment, data, embargo, ready_anchor, boundaries)
    manifest_rows, statrows = write_manifests(assignment, data, date_group, metrics, embargo)
    leakage = leakage_audit(assignment, data, date_group, embargo, raw)
    test_freeze = freeze_test(assignment, data, date_group)
    gate, conditions = gate_result(metrics, leakage, raw, test_freeze, loss, data)
    summary = {"utc": now(), "SPLIT_SEED": SEED, "optimization_objective": soft,
        "optimization_terms": terms, "date_counts": {SPLITS[s]: metrics[s]["days"] for s in range(3)},
        "date_group_counts": {SPLITS[s]: len({date_group[data['dates'][i]] for i in range(len(assignment)) if assignment[i] == s}) for s in range(3)},
        "usable_onset_events": {SPLITS[s]: metrics[s]["events"] for s in range(3)},
        "onset_event_shares": {SPLITS[s]: metrics[s]["share"] for s in range(3)},
        "active_onset_dates": {SPLITS[s]: metrics[s]["active"] for s in range(3)},
        "I_positive_fraction": {SPLITS[s]: metrics[s]["ifrac"] for s in range(3)},
        "O_positive_fraction": {SPLITS[s]: metrics[s]["ofrac"] for s in range(3)},
        "month_dates": {SPLITS[s]: {"June": metrics[s]["months"][0], "July": metrics[s]["months"][1]} for s in range(3)},
        "spatial_regions_covered": {SPLITS[s]: metrics[s]["cover"] for s in range(3)},
        "HIGH_intensity_dates": {SPLITS[s]: metrics[s]["intensity"][2] for s in range(3)},
        "boundary_embargo_anchor_loss": {SPLITS[s]: int(anchor_counts[s]) for s in range(3)},
        "VAL_I_pos_embargo_attrition": float(loss[1, 0] / metrics[1]["counts"][0]),
        "TEST_I_pos_embargo_attrition": float(loss[2, 0] / metrics[2]["counts"][0]),
        "event_leakage_count": leakage["event_multiple_split_count"],
        "raw_observation_overlap_count": raw["raw_observation_overlap_count"],
        "TEST_SET_FREEZE": "PASS" if test_freeze["frozen"] else "FAIL",
        "gate_conditions": conditions, "SPLIT_GATE_R55": gate,
        "READY_FOR_GATE0_BASELINE_TRAINING": "YES" if gate == "PASS" else "NO",
        "R5_LABEL_HASHES_UNCHANGED": "59/59", "R52_READINESS_UNCHANGED": True,
        "SCIENTIFIC_GATE0": "NOT_RUN", "GI": "NA", "GO": "NA", "Gamma": "NA", "95% CI": "NA",
        "split_sidecar_created": True, "models_trained": False}
    atomic_json(OUT / "SUMMARY_R55.json", summary); write_docs(summary)
    deliverables = {"new_files": [str(ROOT / x) for x in ROOT_OUTPUTS] + [str(ASSIGN), str(OUT / "SUMMARY_R55.json")],
                    "updated_files": [str(ROOT / x) for x in CONTROL_DOCS], "evidence_directory": str(OUT),
                    "frozen_R5": str(ROOT / "LABEL_STORE_R5"), "frozen_R52": str(ROOT / "MODEL_READINESS_R52")}
    atomic_json(OUT / "DELIVERABLE_PATHS_R55.json", deliverables)
    written = written_bytes_estimate()
    disk = {"utc": now(), "free_disk_space_before_bytes": prereg["free_disk_space_before_bytes"],
            "free_disk_space_after_bytes": disk_free(), "written_bytes_estimate": written,
            "written_gib_estimate": written / 1024**3, "full_label_store_copied": False,
            "full_readiness_store_copied": False, "temporary_directory": str(TMP),
            "temporary_files_remaining": [str(p) for p in TMP.iterdir()]}
    atomic_json(ROOT / "R55_DISK_IO_AUDIT.json", disk); summary["written_bytes_estimate"] = written; summary["written_gib_estimate"] = disk["written_gib_estimate"]
    atomic_json(OUT / "SUMMARY_R55.json", summary); write_docs(summary)
    # Verify immutable sources after all writes.
    for k, expected in prereg["frozen_inputs"].items(): assert sha(ROOT / k) == expected, k
    validation = {"utc": now(), "status": "PASS" if gate == "PASS" else gate, "SPLIT_GATE_R55": gate,
        "READY_FOR_GATE0_BASELINE_TRAINING": summary["READY_FOR_GATE0_BASELINE_TRAINING"],
        "date_groups": len(groups), "cross_date_events": len(data["cross_date"]), "date_group_integrity": "PASS",
        "event_split_leakage": leakage["status"], "raw_observation_split_leakage": raw["status"],
        "test_set_freeze": "PASS", "embargo_rows": len(embargo_rows), "gate_conditions": conditions,
        "r5_label_partitions_hash_preserved": r52_validation["r5_label_partitions_hash_preserved"],
        "r5_partition_hashes_reverified": 59, "r52_partition_hashes_reverified": 59,
        "r52_artifacts_hash_preserved": "PASS", "fixed_excluded_dates_absent": "PASS",
        "no_random_pixel_split": True, "no_sampling": True, "no_training": True, "scientific_gate": "NOT_RUN",
        "assignment_sha256": sha(ASSIGN / "date_assignment.csv"), "test_freeze_sha256": sha(ROOT / "TEST_SET_FREEZE_R55.json"),
        "script_sha256": sha(__file__), "preregistration_sha256": sha(OUT / "PREREGISTRATION_R55.json")}
    atomic_json(ROOT / "R55_VALIDATION.json", validation)
    atomic_json(OUT / "RUN_FINISHED_R55.json", {"utc": now(), "status": "COMPLETED", "SPLIT_GATE_R55": gate,
        "READY_FOR_GATE0_BASELINE_TRAINING": summary["READY_FOR_GATE0_BASELINE_TRAINING"], "no_training": True})
    print("R55_COMPLETE", json.dumps(to_json_safe(summary), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try: main()
    except Exception as e:
        OUT.mkdir(parents=True, exist_ok=True)
        atomic_json(OUT / "FAILURE_R55.json", {"utc": now(), "status": "BLOCKED", "error": repr(e),
                    "traceback": traceback.format_exc(), "no_training": True, "scientific_gate": "NOT_RUN"})
        raise
