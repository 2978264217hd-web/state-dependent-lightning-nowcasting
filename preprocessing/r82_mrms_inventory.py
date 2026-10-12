"""R82 immutable NOAA MRMS inventory for the frozen three-product data scope.

This script only lists official NODD objects and validates daily cadence.  It
does not download scientific payloads, create labels, or load a model.
"""
import csv
import datetime as dt
import hashlib
import http.client
import json
import os
import tempfile
import time
import traceback
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__import__('os').environ['LIGHTNING_WORKSPACE']).resolve()
OUT = ROOT / "R8_EXTERNAL_CONFIRMATION" / "R82_CONFIRMATORY_DATA_BUILD"
TMP = OUT / "tmp_r82"
HOST = "noaa-mrms-pds.s3.amazonaws.com"
PRODUCTS = ("MergedReflectivityQCComposite", "Reflectivity_-10C", "EchoTop_30")
DAYS = [(dt.date(2025, 6, 1) + dt.timedelta(days=i)).isoformat() for i in range(61)]
UTC = dt.timezone.utc


def now():
    return dt.datetime.now(UTC).isoformat()


def atomic_json(path, obj):
    path = Path(path)
    TMP.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=TMP)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.write("\n")
        json.loads(Path(raw).read_text(encoding="utf-8"))
        path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(raw, path)
    finally:
        if os.path.exists(raw):
            os.unlink(raw)


def atomic_csv(path, rows):
    path = Path(path)
    rows = list(rows)
    fields = ["dataset", "date", "key", "url", "bytes", "start_utc", "etag"]
    TMP.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=TMP)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)
        with open(raw, encoding="utf-8-sig", newline="") as f:
            assert sum(1 for _ in csv.DictReader(f)) == len(rows)
        path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(raw, path)
    finally:
        if os.path.exists(raw):
            os.unlink(raw)


def parse_stamp(key):
    token = key.rsplit("_", 1)[1].split(".", 1)[0]
    return dt.datetime.strptime(token, "%Y%m%d-%H%M%S").replace(tzinfo=UTC)


def request(query):
    last = None
    for attempt in range(1, 6):
        conn = None
        try:
            conn = http.client.HTTPSConnection(HOST, timeout=90)
            conn.request("GET", "/?" + urllib.parse.urlencode(query), headers={"User-Agent": "J1-R3-R82-MRMS-Inventory/1.0"})
            response = conn.getresponse(); body = response.read()
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}")
            return body
        except Exception as exc:
            last = repr(exc)
            if attempt < 5:
                time.sleep(min(30, 2 ** attempt))
        finally:
            if conn is not None:
                conn.close()
    raise RuntimeError(f"MRMS_LIST_RETRIES_EXHAUSTED:{last}")


def list_one(dataset, day):
    prefix = f"CONUS/{dataset}_00.50/{day.replace('-', '')}/"
    token = None; rows = []
    while True:
        query = {"list-type": "2", "prefix": prefix}
        if token: query["continuation-token"] = token
        root = ET.fromstring(request(query))
        ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
        for item in root.findall("s:Contents", ns):
            key = item.findtext("s:Key", namespaces=ns)
            if not key or not key.endswith(".grib2.gz"):
                continue
            stamp = parse_stamp(key)
            assert stamp.date().isoformat() == day
            rows.append({"dataset": dataset, "date": day, "key": key,
                         "url": "https://" + HOST + "/" + key,
                         "bytes": int(item.findtext("s:Size", namespaces=ns)),
                         "start_utc": stamp.isoformat(),
                         "etag": item.findtext("s:ETag", namespaces=ns)})
        truncated = root.findtext("s:IsTruncated", namespaces=ns)
        if str(truncated).lower() == "false":
            break
        token = root.findtext("s:NextContinuationToken", namespaces=ns)
        if not token: raise RuntimeError("Missing S3 continuation token")
    rows.sort(key=lambda r: (r["start_utc"], r["key"]))
    epochs = [dt.datetime.fromisoformat(r["start_utc"]).timestamp() for r in rows]
    gaps = [b-a for a, b in zip(epochs, epochs[1:])]
    return rows, {"dataset": dataset, "date": day, "object_count": len(rows),
                  "first_timestamp": rows[0]["start_utc"] if rows else None,
                  "last_timestamp": rows[-1]["start_utc"] if rows else None,
                  "estimated_cadence_seconds": min(gaps) if gaps else None,
                  "largest_temporal_gap_seconds": max(gaps) if gaps else None,
                  "total_bytes": sum(r["bytes"] for r in rows)}


def main():
    OUT.mkdir(parents=True, exist_ok=True); TMP.mkdir(parents=True, exist_ok=True)
    existing = OUT / "R82_MRMS_OBJECT_INVENTORY.csv"
    rows = []
    daily = []
    for n, day in enumerate(DAYS, 1):
        for product in PRODUCTS:
            listed, summary = list_one(product, day)
            rows.extend(listed); daily.append(summary)
        atomic_json(OUT / "R82_PROGRESS.json", {"utc": now(), "phase": "MRMS_INVENTORY",
                    "date": day, "dates_completed": n, "dates_required": len(DAYS),
                    "objects_listed": len(rows), "payload_downloaded": False,
                    "models_loaded": False})
    atomic_csv(existing, rows)
    atomic_json(OUT / "R82_MRMS_DAILY_AVAILABILITY.json", {"utc": now(), "rows": daily})
    report = {"utc": now(), "status": "PASS", "days": len(DAYS), "products": list(PRODUCTS),
              "objects": len(rows), "bytes": sum(r["bytes"] for r in rows),
              "source": "official NOAA NODD noaa-mrms-pds", "payload_downloaded": False,
              "models_loaded": False,
              "inventory_sha256": hashlib.sha256(existing.read_bytes()).hexdigest()}
    atomic_json(OUT / "R82_MRMS_INVENTORY_COMPLETE.json", report)
    print("R82_MRMS_INVENTORY_COMPLETE", json.dumps(report), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        atomic_json(OUT / "R82_MRMS_INVENTORY_FAILURE.json", {"utc": now(), "status": "BLOCKED",
                    "error": repr(exc), "traceback": traceback.format_exc(), "models_loaded": False})
        raise
