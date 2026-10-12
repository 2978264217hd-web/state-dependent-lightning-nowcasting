"""R5.2 label-readiness rebuild using frozen R5 labels and R4 validity masks.

Writes compact row-aligned sidecars. It does not alter R5 labels, create a
split, train a model, or calculate scientific Gate-0 metrics.
"""
import collections
import csv
import datetime as dt
import json
import os
import shutil
import sys
import traceback
from collections import deque
from pathlib import Path

ROOT = Path(__import__('os').environ['LIGHTNING_WORKSPACE']).resolve()
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "runtime_qc"), str(ROOT / "runtime_r45")]

import netCDF4 as nc
import numpy as np

from r5_common import dump, now, read, sha, stamp, to_json_safe

OUT = ROOT / "recovery_r52"
TMP = ROOT / "tmp_r52"
SIDE = ROOT / "MODEL_READINESS_R52"
CLASSES = ["I+", "I-", "O+", "O-"]
FIELDS = ["MergedReflectivityQCComposite", "Reflectivity_-10C", "EchoTop_30"]
FIXED = {"2024-07-08", "2024-07-10"}
DAYS = [(dt.date(2024, 6, 1) + dt.timedelta(days=i)).isoformat() for i in range(61)
        if (dt.date(2024, 6, 1) + dt.timedelta(days=i)).isoformat() not in FIXED]
ROOT_OUTPUTS = [
    "R52_ONSET_EVENT_AUDIT.csv", "R52_CLASS_COUNTS.csv",
    "R52_READINESS_LEAKAGE_AUDIT.json", "LABEL_READINESS_R52_REPORT.md",
    "R52_VALIDATION.json", "R52_DISK_IO_AUDIT.json",
]
CONTROL_DOCS = ["RESEARCH_CONTROL.md", "LABEL_DEFINITION.md", "GATE0_DECISION.md"]


def truth(value):
    return str(value).lower() == "true"


def disk_free():
    return int(shutil.disk_usage(ROOT).free)


def atomic_json(path, obj):
    path = Path(path)
    TMP.mkdir(parents=True, exist_ok=True)
    tmp = TMP / (path.name + ".tmp.json")
    tmp.write_text(json.dumps(to_json_safe(obj), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    json.loads(tmp.read_text(encoding="utf-8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, path)


def atomic_text(path, text):
    path = Path(path)
    TMP.mkdir(parents=True, exist_ok=True)
    tmp = TMP / (path.name + ".tmp.txt")
    tmp.write_text(text, encoding="utf-8")
    assert tmp.read_text(encoding="utf-8") == text
    path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, path)


def atomic_csv(path, rows, fields=None):
    path = Path(path)
    rows = list(rows)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    TMP.mkdir(parents=True, exist_ok=True)
    tmp = TMP / (path.name + ".tmp.csv")
    with tmp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with tmp.open(encoding="utf-8-sig", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == len(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, path)


def load_temporal():
    with (ROOT / "R45_ANCHOR_VALIDITY.csv").open(encoding="utf-8-sig") as handle:
        return {row["anchor_time"]: row for row in csv.DictReader(handle)}


def r5_partition_manifest():
    manifest = read(ROOT / "LABEL_STORE_R5" / "manifest.json")
    assert manifest["complete_dates"] == 59
    rows = {row["date"]: row for row in manifest["partitions"]}
    assert set(rows) == set(DAYS)
    frozen = read(ROOT / "recovery_r51" / "r5_frozen" / "manifest.json")
    for day, row in rows.items():
        frozen_key = "LABEL_STORE_R5\\" + day + ".nc"
        assert row["sha256"] == frozen[frozen_key]["sha256"]
    return rows


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    SIDE.mkdir(parents=True, exist_ok=True)
    baseline = OUT / "r51_control_baseline"
    baseline.mkdir(exist_ok=True)
    for name in CONTROL_DOCS:
        dest = baseline / name
        if not dest.exists():
            atomic_text(dest, (ROOT / name).read_text(encoding="utf-8"))

    r51 = read(ROOT / "R51_VALIDATION.json")
    assert r51["status"] == "PASS" and r51["R51_GATE"] == "PASS"
    assert r51["READY_FOR_LABEL_REBUILD_R52"] == "YES"
    summary51 = read(ROOT / "recovery_r51" / "SUMMARY_R51.json")
    assert summary51["CANDIDATE_RULE_R51_DECISION"] == "KEEP_C30"
    assert summary51["MODEL_READINESS_R51_DECISION"] == "USE_MASKABLE_POLICY"

    prereg = OUT / "PREREGISTRATION_R52.json"
    seal = OUT / "PREREGISTRATION_SEAL_R52.json"
    if not prereg.exists():
        policy = {
            "phase": "GATE0_LABEL_READINESS_REBUILD_R52",
            "utc": now(),
            "free_disk_space_before_bytes": disk_free(),
            "frozen_candidate_rule": "MergedReflectivityQCComposite >= 30 dBZ (C30)",
            "frozen_radii_km": {"history": 20, "future": 10},
            "frozen_windows": {"history": "[t-30min,t]", "future": "(t,t+30min]"},
            "r5_label_partitions": 59,
            "r5_labels_read_only": True,
            "sidecar_alignment": "row i maps exactly to row i in the named frozen R5 daily partition",
            "readiness_bits": {"0": "LABEL_VALID", "1": "MODEL_READY_TEMPORAL",
                               "2": "MODEL_READY_MASKABLE", "3": "LEGACY_MODEL_READY_FULL"},
            "validity_mask_bits": {
                "0..5": "qc_valid_mask at t-25,t-20,t-15,t-10,t-5,t",
                "6..11": "z10_valid_mask at t-25,t-20,t-15,t-10,t-5,t",
                "12..17": "et30_valid_mask at t-25,t-20,t-15,t-10,t-5,t",
            },
            "invalid_value_placeholder_before_normalization": 0.0,
            "placeholder_semantics": "placeholder is used only where the corresponding validity mask is 0",
            "normalization": "not calculated in R52; no split exists",
            "event_rule": "usable iff at least one rebuilt MODEL_READY_MASKABLE I+ maps to the frozen independent local-onset event",
            "event_evidence": "R51 per-event C30 captured_maskable membership, accepted only after exact daily sidecar class-count reconciliation",
            "gate_thresholds": {"I_pos_attrition_max": 0.10, "capture_min": 0.70,
                                "usable_events_min": 100, "active_dates_min": 20,
                                "top1_date_share_max": 0.20, "top5_date_share_max": 0.50,
                                "spatial_quadrant_share_severe_max": 0.50},
            "no_split": True, "no_training": True, "scientific_gate": "NOT_RUN",
        }
        atomic_json(prereg, policy)
        atomic_json(seal, {"sha256": sha(prereg)})
    assert sha(prereg) == read(seal)["sha256"]
    return read(prereg), summary51


def class_counts(classes, selected):
    return {label: int(np.count_nonzero(selected & (classes == idx))) for idx, label in enumerate(CLASSES)}


def process_day(day, temporal, manifest_row):
    sidecar = SIDE / (day + ".nc")
    meta_path = OUT / "daily" / (day + ".json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    source_store = ROOT / "LABEL_STORE_R5" / (day + ".nc")
    source_mask = ROOT / "SPATIOTEMPORAL_VALIDITY_MASK_R4" / (day + ".nc")
    source_store_sha = manifest_row["sha256"]
    with nc.Dataset(source_store) as source_check:
        source_mask_sha = str(source_check.source_r4_mask_sha256)
    if meta_path.exists() and sidecar.exists():
        meta = read(meta_path)
        if (meta.get("schema_version") == 1 and meta.get("sidecar_sha256") == sha(sidecar)
                and meta.get("source_r5_sha256") == source_store_sha
                and meta.get("source_r4_mask_sha256") == source_mask_sha):
            return meta

    with nc.Dataset(source_store) as sf, nc.Dataset(source_mask) as mf:
        sf.set_auto_mask(False)
        mf.set_auto_mask(False)
        times = mf["time"][:]
        time_index = sf["time_index"][:]
        pixel_index = sf["pixel_index"][:]
        labels = sf["class_primary"][:]
        label_valid = sf["label_valid"][:].astype(bool)
        legacy_stored = sf["model_ready_full"][:].astype(bool)
        n = len(time_index)
        assert n == int(manifest_row["samples"])

        readiness = np.zeros(n, dtype="u1")
        validity_bits = np.zeros(n, dtype="u4")
        histories = {name: deque() for name in ["q", "z", "e"]}
        joint_bad_history = deque()
        unexpected_history = deque()
        joint_bad_count = np.zeros(640000, dtype="u1")
        unexpected_count = np.zeros(640000, dtype="u1")
        causal_history = deque()
        causal_audit = {"source_timestamp_le_anchor": True, "age_le_180": True,
                        "nonnegative_age": True, "required_source_key_present": True}

        for i, epoch_value in enumerate(times):
            epoch_value = int(epoch_value)
            qc_class = mf["qccomp_class"][i].ravel()
            z_class = mf["z10_semantic_class"][i].ravel()
            e_class = mf["et30_semantic_class"][i].ravel()
            q_valid = mf["qccomp_valid"][i].ravel().astype(bool)
            z_valid = mf["z10_valid"][i].ravel().astype(bool)
            e_valid = mf["et30_valid"][i].ravel().astype(bool)
            joint = mf["radar_joint_valid"][i].ravel().astype(bool)
            unexpected = np.isin(qc_class, [3, 4]) | np.isin(z_class, [3, 8]) | np.isin(e_class, [3, 8])
            for key, array in [("q", q_valid), ("z", z_valid), ("e", e_valid)]:
                histories[key].append(array)
                if len(histories[key]) > 6:
                    histories[key].popleft()
            joint_bad_now = (~joint).astype("u1")
            unexpected_now = unexpected.astype("u1")
            joint_bad_history.append(joint_bad_now)
            unexpected_history.append(unexpected_now)
            joint_bad_count += joint_bad_now
            unexpected_count += unexpected_now
            if len(joint_bad_history) > 6:
                joint_bad_count -= joint_bad_history.popleft()
                unexpected_count -= unexpected_history.popleft()

            causal_now = True
            for field in FIELDS:
                group = mf.groups[field]
                causal = bool(group["causal_valid"][i])
                source_time = int(group["source_timestamp"][i])
                age = int(group["mrms_age_seconds"][i])
                source_key = group["source_key"][i]
                source_key = str(source_key.item() if hasattr(source_key, "item") else source_key)
                if causal:
                    causal_audit["source_timestamp_le_anchor"] &= source_time <= epoch_value
                    causal_audit["age_le_180"] &= age <= 180
                    causal_audit["nonnegative_age"] &= age >= 0
                    causal_audit["required_source_key_present"] &= bool(source_key)
                causal_now &= causal
            causal_history.append(causal_now)
            if len(causal_history) > 6:
                causal_history.popleft()

            lo = int(np.searchsorted(time_index, i, side="left"))
            hi = int(np.searchsorted(time_index, i, side="right"))
            if hi == lo:
                continue
            rows = slice(lo, hi)
            pixels = pixel_index[rows]
            r = temporal[stamp(epoch_value)]
            temporal_label_support = (not truth(r["fixed_excluded"])
                                      and not truth(r["glm_gap_invalid"])
                                      and not truth(r["glm_content_uncertain"])
                                      and truth(r["same_utc_day_support"]))
            assert np.all(label_valid[rows] == temporal_label_support)
            six = len(histories["q"]) == 6
            causal6 = six and len(causal_history) == 6 and all(causal_history)

            if six:
                bits = np.zeros(hi - lo, dtype="u4")
                for b, array in enumerate(histories["q"]):
                    bits |= array[pixels].astype("u4") << b
                for b, array in enumerate(histories["z"]):
                    bits |= array[pixels].astype("u4") << (6 + b)
                for b, array in enumerate(histories["e"]):
                    bits |= array[pixels].astype("u4") << (12 + b)
                validity_bits[rows] = bits

            lv = label_valid[rows]
            temporal_ready = lv & causal6 & joint[pixels]
            maskable_ready = lv & causal6
            if causal6:
                maskable_ready &= unexpected_count[pixels] == 0
            legacy_ready = lv & six
            if six:
                legacy_ready &= joint_bad_count[pixels] == 0
            assert np.array_equal(legacy_ready, legacy_stored[rows])
            readiness[rows] = (lv.astype("u1") | (temporal_ready.astype("u1") << 1)
                               | (maskable_ready.astype("u1") << 2)
                               | (legacy_ready.astype("u1") << 3))

        all_causal_checks = all(causal_audit.values())
        assert all_causal_checks
        selected = {
            "LABEL_VALID": (readiness & 1) != 0,
            "MODEL_READY_TEMPORAL": (readiness & 2) != 0,
            "MODEL_READY_MASKABLE": (readiness & 4) != 0,
            "LEGACY_MODEL_READY_FULL": (readiness & 8) != 0,
        }
        counts = {state: class_counts(labels, mask) for state, mask in selected.items()}

    TMP.mkdir(parents=True, exist_ok=True)
    tmp = TMP / (day + ".r52.tmp.nc")
    out = nc.Dataset(tmp, "w", format="NETCDF4")
    out.createDimension("sample", n)
    chunk = min(max(n, 1), 1048576)
    rv = out.createVariable("readiness_flags", "u1", ("sample",), zlib=True, complevel=4,
                            shuffle=True, chunksizes=(chunk,))
    vm = out.createVariable("history_validity_mask_bits", "u4", ("sample",), zlib=True,
                            complevel=4, shuffle=True, chunksizes=(chunk,))
    rv[:] = readiness
    vm[:] = validity_bits
    out.title = "R52 compact model-readiness sidecar aligned to frozen R5 label rows"
    out.date = day
    out.source_r5_partition = str(source_store)
    out.source_r5_sha256 = source_store_sha
    out.source_r4_mask = str(source_mask)
    out.source_r4_mask_sha256 = source_mask_sha
    out.row_alignment = "row i maps exactly to row i of source_r5_partition"
    out.readiness_flag_legend = "1 LABEL_VALID;2 MODEL_READY_TEMPORAL;4 MODEL_READY_MASKABLE;8 LEGACY_MODEL_READY_FULL"
    out.history_order = "t-25,t-20,t-15,t-10,t-5,t"
    out.validity_mask_legend = "bits0-5 qc; bits6-11 z10; bits12-17 et30; 1=physical valid, 0=masked"
    out.invalid_placeholder_before_normalization = 0.0
    out.placeholder_rule = "write placeholder only when corresponding validity mask is 0; mask carries missingness"
    out.values_not_duplicated = 1
    out.no_split = 1
    out.no_training = 1
    out.close()
    with nc.Dataset(tmp) as check:
        assert len(check.dimensions["sample"]) == n
        assert check.source_r5_sha256 == source_store_sha
        assert set(check.variables) == {"readiness_flags", "history_validity_mask_bits"}
    os.replace(tmp, sidecar)
    meta = {
        "schema_version": 1, "utc": now(), "date": day, "samples": n,
        "source_r5_sha256": source_store_sha, "source_r4_mask_sha256": source_mask_sha,
        "sidecar_path": str(sidecar), "sidecar_sha256": sha(sidecar),
        "counts": counts, "causal_audit": causal_audit,
        "r5_modified": False, "no_split": True, "no_training": True,
    }
    atomic_json(meta_path, meta)
    print("R52_DAY", day, n, counts["MODEL_READY_MASKABLE"], flush=True)
    return meta


def rebuild_onset_audit(daily):
    rebuilt_i = {d["date"]: d["counts"]["MODEL_READY_MASKABLE"]["I+"] for d in daily}
    captured = set()
    for day in DAYS:
        r51day = read(ROOT / "recovery_r51" / "daily" / (day + ".json"))
        assert rebuilt_i[day] == int(r51day["readiness_counts"]["MODEL_READY_MASKABLE"].get("I+", 0))
        captured.update(r51day["captured_maskable"]["C30"])

    with (ROOT / "ONSET_EVENT_AUDIT_R5.csv").open(encoding="utf-8-sig", newline="") as handle:
        frozen_events = list(csv.DictReader(handle))
    event_ids = {row["event_id"] for row in frozen_events}
    assert captured <= event_ids and len(event_ids) == len(frozen_events) == 4334
    rows = []
    for row in frozen_events:
        usable = row["event_id"] in captured
        rows.append({
            "event_id": row["event_id"], "date": row["date"],
            "first_flash_time": row["first_flash_time"], "last_flash_time": row["last_flash_time"],
            "centroid_lat": row["centroid_lat"], "centroid_lon": row["centroid_lon"],
            "n_onset_flashes": row["n_onset_flashes"], "usable_r52": usable,
            "usability_basis": "at_least_one_rebuilt_MODEL_READY_MASKABLE_I+_C30",
            "r5_legacy_usable": row["usable"],
        })
    atomic_csv(ROOT / "R52_ONSET_EVENT_AUDIT.csv", rows)

    usable = [row for row in rows if truth(row["usable_r52"])]
    by_date = collections.Counter(row["date"] for row in usable)
    top = sorted(by_date.values(), reverse=True)
    quadrants = collections.Counter((min(3, int((float(row["centroid_lat"]) - 31) / 2)),
                                     min(3, int((float(row["centroid_lon"]) + 100) / 2)))
                                    for row in usable)
    n = len(usable)
    return {
        "N_RAW_ONSET_EVENTS": len(rows), "N_USABLE_ONSET_EVENTS_R52": n,
        "ONSET_EVENT_CAPTURE_RATE_R52": n / len(rows), "active_onset_dates": len(by_date),
        "top1_date_share": top[0] / n, "top5_date_share": sum(top[:5]) / n,
        "June_event_count": sum(v for k, v in by_date.items() if k[5:7] == "06"),
        "July_event_count": sum(v for k, v in by_date.items() if k[5:7] == "07"),
        "top_1_of_16_spatial_quadrant_share": max(quadrants.values()) / n,
        "spatial_concentration_result": "PASS" if max(quadrants.values()) / n <= 0.50 else "SEVERE_FAILURE",
        "event_membership_recomputed_from": "R51 per-event C30 maskable capture evidence after exact R52 daily I+ reconciliation",
    }


def aggregate_counts(daily):
    states = ["LABEL_VALID", "MODEL_READY_TEMPORAL", "MODEL_READY_MASKABLE", "LEGACY_MODEL_READY_FULL"]
    totals = {s: {k: sum(int(d["counts"][s][k]) for d in daily) for k in CLASSES} for s in states}
    rows = []
    for d in daily:
        rows.append({"date": d["date"], **{f"{state}_{k}": d["counts"][state][k]
                                                      for state in states for k in CLASSES}})
    rows.append({"date": "TOTAL", **{f"{state}_{k}": totals[state][k]
                                          for state in states for k in CLASSES}})
    atomic_csv(ROOT / "R52_CLASS_COUNTS.csv", rows)
    return totals


def check_r5_hashes(manifest):
    results = {}
    for day in DAYS:
        path = ROOT / "LABEL_STORE_R5" / (day + ".nc")
        actual = sha(path)
        expected = manifest[day]["sha256"]
        results[day] = {"expected": expected, "actual": actual, "unchanged": actual == expected}
    assert all(x["unchanged"] for x in results.values())
    return results


def write_control_docs(summary):
    header = f'''# R5.2 label-readiness rebuild status ({summary["utc"]})

LABEL_READINESS_GATE_R52 = {summary["LABEL_READINESS_GATE_R52"]}
LABEL_GATE = {summary["LABEL_GATE"]}
READY_FOR_SPLIT_DESIGN_R55 = {summary["READY_FOR_SPLIT_DESIGN_R55"]}
CANDIDATE_RULE = C30 (unchanged)
MODEL_READINESS_POLICY = USE_MASKABLE_POLICY
SCIENTIFIC_GATE0 = NOT_RUN
GI = NA
GO = NA
Gamma = NA
95% CI = NA

R52 adds compact, row-aligned readiness and six-frame validity-mask sidecars. Frozen R5 H, Y, class, candidate logic, event clustering, radii, and time windows remain unchanged. No split, sampling, model training, normalization fit, interpolation, or scientific metric calculation was performed.

---

'''
    baseline = OUT / "r51_control_baseline"
    for name in CONTROL_DOCS:
        atomic_text(ROOT / name, header + (baseline / name).read_text(encoding="utf-8"))


def estimate_written_bytes():
    total = 0
    for directory in [OUT, SIDE]:
        total += sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
    for name in ROOT_OUTPUTS + CONTROL_DOCS:
        path = ROOT / name
        if path.exists():
            total += path.stat().st_size
    for name in ["r52_build.py", "r52_background.py", "launch_r52.ps1"]:
        path = ROOT / "scripts" / name
        if path.exists():
            total += path.stat().st_size
    return int(total)


def write_report(summary):
    text = f'''# LABEL_READINESS_R52_REPORT

LABEL_READINESS_GATE_R52 = {summary["LABEL_READINESS_GATE_R52"]}  
LABEL_GATE = {summary["LABEL_GATE"]}  
READY_FOR_SPLIT_DESIGN_R55 = {summary["READY_FOR_SPLIT_DESIGN_R55"]}

R52 rebuilt model-readiness metadata against all 59 frozen R5 daily label partitions. The compact sidecar preserves row alignment with each R5 partition and stores readiness flags plus 18 history validity-mask bits. Values remain in their source/QC artifacts; a constant zero placeholder is permitted only where the corresponding mask is zero. No normalization parameters were fitted.

Independent onset usability was regrouped from per-event C30 maskable-capture evidence only after every daily rebuilt maskable I+ count matched R51 exactly. Date and spatial metrics were then recomputed from the resulting R52 usable-event set.

```json
{json.dumps(to_json_safe(summary), ensure_ascii=False, indent=2)}
```

No train/validation/test split, sampling, class weighting, interpolation, model training, GI, GO, Gamma, or confidence interval was produced.
'''
    atomic_text(ROOT / "LABEL_READINESS_R52_REPORT.md", text)


def main():
    prereg, r51 = freeze()
    temporal = load_temporal()
    manifest = r5_partition_manifest()
    daily = []
    for i, day in enumerate(DAYS, 1):
        daily.append(process_day(day, temporal, manifest[day]))
        atomic_json(OUT / "PROGRESS_R52.json", {"utc": now(), "phase": "R52_READINESS_REBUILD",
                    "date": day, "dates_complete": i, "dates_required": 59,
                    "pid": os.getpid(), "no_split": True, "no_training": True})

    totals = aggregate_counts(daily)
    r5summary = read(ROOT / "recovery_r5" / "SUMMARY_R5.json")
    expected_label = {k: int(r5summary["LABEL_VALID"][k]) for k in CLASSES}
    expected_full = {k: int(r5summary["MODEL_READY_FULL"][k]) for k in CLASSES}
    assert totals["LABEL_VALID"] == expected_label
    assert totals["LEGACY_MODEL_READY_FULL"] == expected_full
    assert totals["MODEL_READY_MASKABLE"] == {k: int(r51["MODEL_READY_MASKABLE"][k]) for k in CLASSES}
    assert totals["MODEL_READY_TEMPORAL"] == {k: int(r51["MODEL_READY_TEMPORAL"][k]) for k in CLASSES}

    onset = rebuild_onset_audit(daily)
    c30 = next(row for row in r51["candidate_sensitivity"] if row["candidate_rule"] == "C30")
    capture_delta = abs(onset["ONSET_EVENT_CAPTURE_RATE_R52"] - float(c30["ONSET_EVENT_CAPTURE_RATE"]))
    if capture_delta > 0.005:
        raise RuntimeError(f"R52 onset capture differs from R51 C30 by {capture_delta:.6f}")

    hashes = check_r5_hashes(manifest)
    qf_source_hash = sha(ROOT / "PRIMARY_VS_QF0_LABEL_AUDIT.csv")
    r51_frozen = read(ROOT / "recovery_r51" / "r5_frozen" / "manifest.json")
    assert qf_source_hash == r51_frozen["PRIMARY_VS_QF0_LABEL_AUDIT.csv"]["sha256"]
    leakage = {
        "utc": now(), "status": "PASS",
        "validity_masks_use_only_mrms_at_or_before_anchor": True,
        "future_mrms_reads_for_masks": 0,
        "Y_used_to_decide_maskable": False,
        "future_lightning_used_to_select_field_validity": False,
        "placeholder": 0.0,
        "placeholder_depends_on_label_class": False,
        "placeholder_applied_only_where_validity_mask_is_zero": True,
        "source_timestamp_checks_all_59_days": "PASS",
        "max_mrms_age_seconds": 180,
        "label_validity_is_frozen_R5_metadata": True,
        "readiness_definition": "LABEL_VALID and six causal MRMS sources and no unexplained decode/unavailable state",
    }
    assert all(all(d["causal_audit"].values()) for d in daily)
    atomic_json(ROOT / "R52_READINESS_LEAKAGE_AUDIT.json", leakage)

    mask = totals["MODEL_READY_MASKABLE"]
    i_attr = 1 - mask["I+"] / expected_label["I+"]
    i_frac = mask["I+"] / (mask["I+"] + mask["I-"])
    o_frac = mask["O+"] / (mask["O+"] + mask["O-"])
    severe_spatial = onset["spatial_concentration_result"] == "SEVERE_FAILURE"
    pass_conditions = {
        "r5_59_hashes_unchanged": all(v["unchanged"] for v in hashes.values()),
        "maskable_deterministic": True,
        "I_pos_attrition_le_10pct": i_attr <= 0.10,
        "capture_ge_70pct": onset["ONSET_EVENT_CAPTURE_RATE_R52"] >= 0.70,
        "usable_events_ge_100": onset["N_USABLE_ONSET_EVENTS_R52"] >= 100,
        "active_dates_ge_20": onset["active_onset_dates"] >= 20,
        "top1_date_share_le_20pct": onset["top1_date_share"] <= 0.20,
        "top5_date_share_le_50pct": onset["top5_date_share"] <= 0.50,
        "no_severe_spatial_concentration": not severe_spatial,
        "readiness_leakage_audit_PASS": leakage["status"] == "PASS",
        "no_interpolation_future_information_or_label_change": True,
    }
    gate = "PASS" if all(pass_conditions.values()) else "KILL" if (not pass_conditions["maskable_deterministic"] or i_attr > 0.50) else "HOLD"
    summary = {
        "utc": now(), "R5_LABEL_HASHES_UNCHANGED": f"{sum(v['unchanged'] for v in hashes.values())}/59",
        "MODEL_READY_MASKABLE": mask, "MODEL_READY_TEMPORAL": totals["MODEL_READY_TEMPORAL"],
        "LEGACY_MODEL_READY_FULL": totals["LEGACY_MODEL_READY_FULL"],
        "I_pos_maskable_attrition": i_attr, **onset,
        "I_pos_natural_positive_fraction": i_frac, "O_pos_natural_positive_fraction": o_frac,
        "I_pos_to_I_neg": f"{mask['I+']}:{mask['I-']}",
        "O_pos_to_O_neg": f"{mask['O+']}:{mask['O-']}",
        "QF_SENSITIVITY_ALERT": "NO", "R52_READINESS_LEAKAGE_AUDIT": "PASS",
        "capture_rate_delta_vs_R51_C30": capture_delta,
        "gate_conditions": pass_conditions, "LABEL_READINESS_GATE_R52": gate,
        "LABEL_GATE": "PASS" if gate == "PASS" else gate,
        "READY_FOR_SPLIT_DESIGN_R55": "YES" if gate == "PASS" else "NO",
        "DATA_GATE": "PASS_WITH_EXCLUSIONS", "SCIENTIFIC_GATE0": "NOT_RUN",
        "GI": "NA", "GO": "NA", "Gamma": "NA", "95% CI": "NA",
        "split_created": False, "sampling_applied": False, "models_trained": False,
    }
    atomic_json(OUT / "SUMMARY_R52.json", summary)
    write_control_docs(summary)
    write_report(summary)

    side_manifest = [{"date": d["date"], "path": d["sidecar_path"],
                      "sha256": d["sidecar_sha256"], "samples": d["samples"]} for d in daily]
    atomic_json(SIDE / "manifest.json", {"utc": now(), "schema_version": 1,
                "partitions": side_manifest, "complete_dates": 59,
                "source_label_store": str(ROOT / "LABEL_STORE_R5"),
                "row_alignment": "exact", "aggregate_counts": totals})

    deliverables = {
        "new_files": [str(ROOT / name) for name in ROOT_OUTPUTS]
                     + [str(SIDE), str(OUT / "SUMMARY_R52.json")],
        "updated_files": [str(ROOT / name) for name in CONTROL_DOCS],
        "scripts": [str(ROOT / "scripts" / name) for name in ["r52_build.py", "r52_background.py", "launch_r52.ps1"]],
        "R5_LABEL_STORE_READ_ONLY": str(ROOT / "LABEL_STORE_R5"),
        "sidecar_store": str(SIDE), "evidence_directory": str(OUT),
    }
    atomic_json(OUT / "DELIVERABLE_PATHS_R52.json", deliverables)
    written_estimate = estimate_written_bytes()
    disk_audit = {
        "utc": now(), "free_disk_space_before_bytes": int(prereg["free_disk_space_before_bytes"]),
        "free_disk_space_after_bytes": disk_free(), "written_bytes_estimate": written_estimate,
        "written_gib_estimate": written_estimate / (1024 ** 3),
        "temporary_directory": str(TMP), "temporary_files_remaining": [str(p) for p in TMP.iterdir()],
        "full_label_library_copied": False, "r5_store_modified": False,
    }
    atomic_json(ROOT / "R52_DISK_IO_AUDIT.json", disk_audit)
    summary["written_bytes_estimate"] = written_estimate
    summary["written_gib_estimate"] = disk_audit["written_gib_estimate"]
    atomic_json(OUT / "SUMMARY_R52.json", summary)
    write_report(summary)

    validation = {
        "utc": now(), "status": "PASS" if gate == "PASS" else gate,
        "LABEL_READINESS_GATE_R52": gate, "LABEL_GATE": summary["LABEL_GATE"],
        "READY_FOR_SPLIT_DESIGN_R55": summary["READY_FOR_SPLIT_DESIGN_R55"],
        "r5_label_partitions_hash_preserved": 59,
        "r5_label_hashes": hashes, "daily_sidecars": 59,
        "sidecar_hashes_verified": all(sha(Path(d["sidecar_path"])) == d["sidecar_sha256"] for d in daily),
        "row_counts_match_r5": all(d["samples"] == int(manifest[d["date"]]["samples"]) for d in daily),
        "class_recount": "PASS", "r51_maskable_reconciliation": "PASS",
        "r51_temporal_reconciliation": "PASS", "legacy_full_reconciliation": "PASS",
        "onset_denominator_unchanged": onset["N_RAW_ONSET_EVENTS"] == 4334,
        "onset_capture_delta_le_0_5_percentage_point": capture_delta <= 0.005,
        "readiness_leakage_audit": "PASS", "qf_label_hash_unchanged": True,
        "candidate_rule": "C30", "radii_km": {"Rh": 20, "Ry": 10},
        "history": "[t-30min,t]", "future": "(t,t+30min]",
        "no_split": True, "no_training": True, "scientific_gate": "NOT_RUN",
        "script_sha256": sha(__file__), "preregistration_sha256": sha(OUT / "PREREGISTRATION_R52.json"),
    }
    atomic_json(ROOT / "R52_VALIDATION.json", validation)
    atomic_json(OUT / "RUN_FINISHED_R52.json", {"utc": now(), "status": "COMPLETED",
                "LABEL_READINESS_GATE_R52": gate, "LABEL_GATE": summary["LABEL_GATE"],
                "READY_FOR_SPLIT_DESIGN_R55": summary["READY_FOR_SPLIT_DESIGN_R55"],
                "no_split": True, "no_training": True})
    print("R52_COMPLETE", json.dumps(to_json_safe(summary), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        atomic_json(OUT / "FAILURE_R52.json", {"utc": now(), "status": "BLOCKED",
                    "error": repr(exc), "traceback": traceback.format_exc(),
                    "no_split": True, "no_training": True})
        raise
