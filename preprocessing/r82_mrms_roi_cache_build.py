import bisect
import concurrent.futures
import datetime as dt
import gzip
import hashlib
import http.client
import json
import os
import tempfile
import time
import traceback
import urllib.parse
from collections import defaultdict
from pathlib import Path

ROOT = Path(__import__('os').environ['LIGHTNING_WORKSPACE']).resolve()
import eccodes as ec
import numpy as np

BASE = ROOT / "R8_EXTERNAL_CONFIRMATION"
OUT = BASE / "R82_CONFIRMATORY_DATA_BUILD"
QC = OUT / "mrms_content_qc"
INV = OUT / "R82_MRMS_OBJECT_INVENTORY.csv"
CACHE = OUT / "mrms_roi_cache"
TMP = OUT / "tmp_r82"
PRODUCTS = ("MergedReflectivityQCComposite", "Reflectivity_-10C", "EchoTop_30")
SPECS = {
    "MergedReflectivityQCComposite": (10, 0, -99, -999),
    "Reflectivity_-10C": (3, 54, -99, -999),
    "EchoTop_30": (3, 45, -1, -3),
}
DAYS = [(dt.date(2025, 6, 1) + dt.timedelta(days=i)).isoformat() for i in range(61)]
UTC = dt.timezone.utc

def now():
    return dt.datetime.now(UTC).isoformat()

def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=TMP)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.write("\n")
        json.loads(Path(tmp).read_text(encoding="utf-8"))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def sha_key(key):
    return hashlib.sha256(key.encode()).hexdigest()

def download(row):
    url = urllib.parse.urlsplit(row["url"])
    expected_bytes = int(row["bytes"])
    last = None
    for attempt in range(1, 6):
        conn = None
        try:
            conn = http.client.HTTPSConnection(url.hostname, timeout=30)
            conn.request("GET", url.path, headers={"User-Agent": "J1-R3-R82-MRMS-ROI-CACHE/1.0"})
            response = conn.getresponse()
            if response.status != 200:
                raise RuntimeError("HTTP_STATUS_" + str(response.status))
            chunks = []
            n = 0
            while True:
                block = response.read(65536)
                if not block:
                    break
                chunks.append(block)
                n += len(block)
            body = b"".join(chunks)
            if n != expected_bytes:
                raise RuntimeError(f"SOURCE_BYTE_COUNT:{n}!={expected_bytes}")
            return body, hashlib.sha256(body).hexdigest()
        except Exception as exc:
            last = repr(exc)
            if attempt < 5:
                time.sleep(min(30, 2 ** attempt))
        finally:
            if conn:
                conn.close()
    raise RuntimeError("R82_CACHE_DOWNLOAD_RETRIES_EXHAUSTED:" + row["key"] + ":" + str(last))

def decode_and_cache(row, expected, cache_dir):
    key = row["key"]
    stem = sha_key(key)
    dest = cache_dir / (stem + ".npz")
    meta_path = cache_dir / (stem + ".json")
    if dest.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (meta.get("key") == key and meta.get("roi_sha256") == expected["roi_sha256"]
                and meta.get("npz_sha256") == sha_file(dest)):
            return {"key": key, "status": "REUSED_VERIFIED", "npz_sha256": meta["npz_sha256"],
                    "download_bytes": 0}
    body, payload_sha = download(row)
    raw = gzip.decompress(body)
    if raw[:4] != b"GRIB" or raw[-4:] != b"7777" or int.from_bytes(raw[8:16], "big") != len(raw):
        raise RuntimeError("GRIB_CONTAINER_INVALID:" + key)
    handle = None
    try:
        handle = ec.codes_new_from_message(raw)
        dataset = row["dataset"]
        category, number, missing, no_coverage = SPECS[dataset]
        if (ec.codes_get(handle, "discipline"), ec.codes_get(handle, "parameterCategory"),
                ec.codes_get(handle, "parameterNumber")) != (209, category, number):
            raise RuntimeError("R82_NEW_SCHEMA_ALERT_PARAMETER:" + key)
        ni, nj = ec.codes_get(handle, "Ni"), ec.codes_get(handle, "Nj")
        if (ni, nj) != (7000, 3500):
            raise RuntimeError("R82_NEW_SCHEMA_ALERT_GRID:" + key)
        lat0 = ec.codes_get(handle, "latitudeOfFirstGridPointInDegrees")
        lon0 = ec.codes_get(handle, "longitudeOfFirstGridPointInDegrees")
        di = ec.codes_get(handle, "iDirectionIncrementInDegrees")
        dj = ec.codes_get(handle, "jDirectionIncrementInDegrees")
        lat = lat0 + np.arange(nj) * dj * (1 if ec.codes_get(handle, "jScansPositively") else -1)
        lon = (lon0 + np.arange(ni) * di * (-1 if ec.codes_get(handle, "iScansNegatively") else 1) + 180) % 360 - 180
        yi = np.flatnonzero((lat >= 31) & (lat <= 39))
        xi = np.flatnonzero((lon >= -100) & (lon <= -92))
        if len(yi) != 800 or len(xi) != 800:
            raise RuntimeError("R82_NEW_SCHEMA_ALERT_ROI_GRID:" + key)
        values = ec.codes_get_array(handle, "values").reshape(nj, ni)[np.ix_(yi, xi)]
        raw_sha = hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
        if raw_sha != expected["roi_sha256"]:
            raise RuntimeError("R82_SOURCE_ROI_HASH_MISMATCH:" + key)
        base = np.zeros(values.shape, dtype="u1")
        base[values == missing] = 1
        base[values == no_coverage] = 2
        base[(~np.isfinite(values)) | (values == ec.codes_get(handle, "missingValue"))] = 3
        counts = {str(i): int(np.count_nonzero(base == i)) for i in range(4)}
        if counts != expected["base_counts"]:
            raise RuntimeError("R82_BASE_CLASS_COUNT_MISMATCH:" + key)
        if dt.datetime(*(ec.codes_get(handle, k) for k in ("year", "month", "day", "hour", "minute", "second")),
                       tzinfo=UTC).isoformat() != row["start_utc"]:
            raise RuntimeError("R82_SOURCE_TIMESTAMP_MISMATCH:" + key)
        lat_roi = lat[yi].astype("f8")
        lon_roi = lon[xi].astype("f8")
        fd, tmp_name = tempfile.mkstemp(prefix=stem + ".", suffix=".tmp.npz", dir=cache_dir)
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            np.savez_compressed(tmp_path, value_raw=values, base_class=base, lat=lat_roi, lon=lon_roi)
            with np.load(tmp_path) as check:
                if check["value_raw"].shape != (800, 800) or check["base_class"].shape != (800, 800):
                    raise RuntimeError("R82_CACHE_PARTITION_SHAPE_INVALID:" + key)
                if hashlib.sha256(np.ascontiguousarray(check["value_raw"]).tobytes()).hexdigest() != raw_sha:
                    raise RuntimeError("R82_CACHE_PARTITION_HASH_INVALID:" + key)
            os.replace(tmp_path, dest)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        meta = {"key": key, "dataset": dataset, "date": row["date"], "start_utc": row["start_utc"],
                "source_bytes": int(row["bytes"]), "payload_sha256": payload_sha, "roi_sha256": raw_sha,
                "base_counts": counts, "npz_sha256": sha_file(dest), "cache_schema": "R82_ROI_RAW_BASE_V1",
                "method": "Frozen R4/R45 ROI decode and sentinel classes; exact prior QC ROI checksum"}
        tmp_meta = meta_path.with_suffix(".tmp.json")
        tmp_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        json.loads(tmp_meta.read_text(encoding="utf-8"))
        os.replace(tmp_meta, meta_path)
        return {"key": key, "status": "CACHED_VERIFIED", "npz_sha256": meta["npz_sha256"],
                "download_bytes": len(body)}
    finally:
        if handle is not None:
            ec.codes_release(handle)

def main():
    complete = json.loads((OUT / "R82_MRMS_CONTENT_QC_COMPLETE.json").read_text(encoding="utf-8"))
    if complete.get("status") != "PASS" or complete.get("dates") != 61:
        raise RuntimeError("R82_MRMS_CONTENT_QC_NOT_COMPLETE")
    if len(list((OUT / "flash_daily").glob("*.npz"))) != 61:
        raise RuntimeError("R82_GLM_DAILY_PARTITIONS_NOT_COMPLETE")
    if len(list(QC.glob("*.json"))) != 61:
        raise RuntimeError("R82_MRMS_QC_DAILY_PARTITIONS_NOT_COMPLETE")
    inventory = {}
    import csv
    with INV.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            row["bytes"] = int(row["bytes"])
            inventory[row["key"]] = row
    selected = defaultdict(dict)
    for day in DAYS:
        report = json.loads((QC / (day + ".json")).read_text(encoding="utf-8"))
        if report.get("status") != "PASS" or report.get("date") != day or report.get("anchors_examined") != 287:
            raise RuntimeError("R82_DAILY_QC_CHECKPOINT_INVALID:" + day)
        if report.get("max_mrms_age_seconds") != 180 or report.get("no_future_source") is not True:
            raise RuntimeError("R82_FROZEN_CAUSAL_RULE_MISMATCH:" + day)
        for item in report["records"]:
            key = item["key"]
            if key not in inventory:
                raise RuntimeError("R82_SELECTED_OBJECT_MISSING_FROM_INVENTORY:" + key)
            row = inventory[key]
            if row["dataset"] != item["dataset"] or row["start_utc"] != item["start_utc"] or row["bytes"] != int(item["bytes"]):
                raise RuntimeError("R82_SELECTED_OBJECT_INVENTORY_MISMATCH:" + key)
            expected = {"roi_sha256": item["roi_sha256"], "base_counts": {str(k): int(v) for k, v in item["base_counts"].items()}}
            if key in selected[day] and selected[day][key]["expected"] != expected:
                raise RuntimeError("R82_INCONSISTENT_QC_DUPLICATE:" + key)
            selected[day][key] = {"row": row, "expected": expected}
    all_selected = {}
    for day in DAYS:
        for key, item in selected[day].items():
            if key in all_selected and all_selected[key]["expected"] != item["expected"]:
                raise RuntimeError("R82_CROSS_DAY_QC_DUPLICATE_MISMATCH:" + key)
            all_selected[key] = item
    if len(all_selected) != 52500:
        raise RuntimeError(f"R82_SELECTED_OBJECT_COUNT_UNEXPECTED:{len(all_selected)}")
    CACHE.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    required_bytes = sum(v["row"]["bytes"] for v in all_selected.values())
    free_before = __import__("shutil").disk_usage(BASE).free
    results = []
    errors = []
    completed = 0
    downloaded = 0
    pending = []
    for key, item in all_selected.items():
        path = CACHE / (sha_key(key) + ".npz")
        meta = CACHE / (sha_key(key) + ".json")
        if path.exists() and meta.exists():
            try:
                m = json.loads(meta.read_text(encoding="utf-8"))
                if m.get("key") == key and m.get("roi_sha256") == item["expected"]["roi_sha256"] and m.get("npz_sha256") == sha_file(path):
                    completed += 1
                    continue
            except Exception:
                pass
        pending.append((key, item))
    state_path = OUT / "R82_MRMS_ROI_CACHE_PROGRESS.json"
    atomic_json(state_path, {"utc": now(), "phase": "MRMS_ROI_CACHE_BUILD", "status": "RUNNING",
                             "selected_object_count": len(all_selected), "verified_cache_objects": completed,
                             "objects_pending": len(pending), "selected_source_bytes": required_bytes,
                             "disk_free_before_bytes": free_before, "worker_count": 2,
                             "transport": "30-second bounded reads; up to five attempts",
                             "labels_constructed": False, "models_loaded": False})
    current_key = None
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
            futures = {}
            for key, item in pending:
                current_key = key
                futures[pool.submit(decode_and_cache, item["row"], item["expected"], CACHE)] = key
                while len(futures) >= 6:
                    done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        key_done = futures.pop(future)
                        result = future.result()
                        completed += 1
                        downloaded += result["download_bytes"]
                        if completed % 20 == 0 or completed == len(all_selected):
                            atomic_json(state_path, {"utc": now(), "phase": "MRMS_ROI_CACHE_BUILD", "status": "RUNNING",
                                                     "selected_object_count": len(all_selected), "verified_cache_objects": completed,
                                                     "objects_pending": len(all_selected)-completed, "download_bytes_this_run": downloaded,
                                                     "selected_source_bytes": required_bytes, "current_key": key_done,
                                                     "disk_free_bytes": __import__("shutil").disk_usage(BASE).free,
                                                     "labels_constructed": False, "models_loaded": False})
            for future in concurrent.futures.as_completed(futures):
                key_done = futures[future]
                result = future.result()
                completed += 1
                downloaded += result["download_bytes"]
                if completed % 20 == 0 or completed == len(all_selected):
                    atomic_json(state_path, {"utc": now(), "phase": "MRMS_ROI_CACHE_BUILD", "status": "RUNNING",
                                             "selected_object_count": len(all_selected), "verified_cache_objects": completed,
                                             "objects_pending": len(all_selected)-completed, "download_bytes_this_run": downloaded,
                                             "selected_source_bytes": required_bytes, "current_key": key_done,
                                             "disk_free_bytes": __import__("shutil").disk_usage(BASE).free,
                                             "labels_constructed": False, "models_loaded": False})
        if completed != len(all_selected):
            raise RuntimeError(f"R82_CACHE_OBJECT_COUNT_MISMATCH:{completed}")
        atomic_json(OUT / "R82_MRMS_ROI_CACHE_COMPLETE.json", {"utc": now(), "status": "PASS",
                    "selected_object_count": len(all_selected), "selected_source_bytes_referenced": required_bytes,
                    "download_bytes_this_run": downloaded, "cache_directory": str(CACHE),
                    "disk_free_before_bytes": free_before, "disk_free_after_bytes": __import__("shutil").disk_usage(BASE).free,
                    "labels_constructed": False, "models_loaded": False})
        atomic_json(state_path, {"utc": now(), "phase": "MRMS_ROI_CACHE_BUILD", "status": "PASS",
                                 "selected_object_count": len(all_selected), "verified_cache_objects": completed,
                                 "objects_pending": 0, "download_bytes_this_run": downloaded,
                                 "disk_free_after_bytes": __import__("shutil").disk_usage(BASE).free,
                                 "labels_constructed": False, "models_loaded": False})
    except Exception as exc:
        schema_alert = "R82_NEW_SCHEMA_ALERT" in repr(exc)
        prefix = "R82_NEW_SCHEMA_ALERT_" if schema_alert else "R82_MRMS_ROI_CACHE_FAILURE_"
        failure = OUT / (prefix + dt.datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + ".json")
        atomic_json(failure, {"utc": now(), "status": "BLOCKED_SCHEMA_CHANGE" if schema_alert else "BLOCKED_ENGINEERING", "phase": "MRMS_ROI_CACHE_BUILD",
                              "current_key": current_key, "verified_cache_objects": completed,
                              "download_bytes_this_run": downloaded, "error": repr(exc),
                              "traceback": traceback.format_exc(), "labels_constructed": False, "models_loaded": False})
        raise

if __name__ == "__main__":
    main()
