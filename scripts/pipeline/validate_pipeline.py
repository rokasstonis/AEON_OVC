"""Validation script for the foragingOVC pipeline (v0.4.0, DJ 2.x).

Checks that all pipeline steps completed correctly by verifying table contents.

Usage:
  python scripts/pipeline/validate_pipeline.py          # Fast checks (structure + metadata)
  python scripts/pipeline/validate_pipeline.py --full   # Also populate & check stream data (slow)

Prerequisites:
  - scripts/pipeline/setup_pipeline.py has been run successfully
  - Credentials configured (see docs/SETUP_GUIDE.md section 3)
"""

import argparse
import json
import sys

# ---------------------------------------------------------------------------
# Experiment configuration (must match setup_pipeline.py)
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "ProjectAeonOVC-2026-09-14_001"  # Must match setup_pipeline.py
DEVICES_SCHEMA = "swc.aeon_exp.foragingOVC.experiment:Experiment"
ARENA_NAME = "circle-2m"
STREAM_POPULATE_LIMIT = 10

SUPPRESS_ERRORS = False


# ---------------------------------------------------------------------------
# Safety: verify DB prefix before ANY pipeline imports
# ---------------------------------------------------------------------------
def verify_config_or_exit():
    import datajoint as dj

    if not dj.config.database.database_prefix:
        print(f"\n  ✗ SAFETY CHECK FAILED: no database prefix configured")
        print(f"    Set database_prefix in datajoint.json")
        sys.exit(1)

    host = dj.config.database.host or ""
    if "aeon-db2" in host:
        print(f"\n  ✗ SAFETY CHECK FAILED: connecting to production host '{host}'")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Check tracking
# ---------------------------------------------------------------------------
class CheckResult:
    def __init__(self):
        self.checks = []

    def record(self, group, name, passed, detail=""):
        self.checks.append({
            "group": group,
            "name": name,
            "passed": passed,
            "detail": detail,
        })
        status = "PASS" if passed else "FAIL"
        detail_str = f" ({detail})" if detail else ""
        print(f"  [{status}] {name}{detail_str}")

    def summary(self):
        total = len(self.checks)
        passed = sum(1 for c in self.checks if c["passed"])
        failed = total - passed

        print(f"\n{'='*60}")
        print(f"  Summary: {passed}/{total} checks passed", end="")
        if failed > 0:
            print(f", {failed} FAILED")
            print()
            for c in self.checks:
                if not c["passed"]:
                    detail_str = f" — {c['detail']}" if c['detail'] else ""
                    print(f"    ✗ [{c['group']}] {c['name']}{detail_str}")
        else:
            print()
        print(f"{'='*60}")

        return failed == 0


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------
def check_catalog(results):
    print(f"\n--- Step 1: Catalog Population ---\n")

    from aeon.dj_pipeline.streams import DeviceType, StreamType

    stream_types = StreamType.to_dicts()
    results.record("Catalog", "StreamType populated", len(stream_types) > 0,
                    f"{len(stream_types)} entries")

    stream_type_names = [st["stream_type"] for st in stream_types]
    results.record("Catalog", "StreamType contains 'Video'", "Video" in stream_type_names)

    device_types = DeviceType.to_arrays("device_type")
    results.record("Catalog", "DeviceType populated", len(device_types) > 0,
                    f"{len(device_types)} entries")
    results.record("Catalog", "DeviceType contains 'Camera'",
                    "Camera" in device_types)

    device_streams = DeviceType.Stream.to_dicts()
    results.record("Catalog", "DeviceType.Stream linked", len(device_streams) > 0,
                    f"{len(device_streams)} entries")


def check_tables(results):
    print(f"\n--- Step 2: Table Creation ---\n")

    import datajoint as dj
    from aeon.dj_pipeline import streams

    dj_tables = []
    for name in dir(streams):
        if name.startswith("_"):
            continue
        obj = getattr(streams, name, None)
        if isinstance(obj, type) and issubclass(obj, dj.Table) and obj is not dj.Table:
            dj_tables.append(name)

    results.record("Tables", "DJ tables in streams module", len(dj_tables) > 0,
                    f"{len(dj_tables)} tables")

    video_tables = [name for name in dj_tables if "Video" in name]
    results.record("Tables", "Video stream tables exist", len(video_tables) > 0,
                    f"{video_tables}")

    results.record("Tables", "Camera table exists",
                    hasattr(streams, "Camera"),
                    "from hasattr check")


def check_experiment(results):
    print(f"\n--- Experiment Registration ---\n")

    from aeon.dj_pipeline import acquisition

    exp_entries = (acquisition.Experiment & {"experiment_name": EXPERIMENT_NAME}).to_dicts()
    results.record("Experiment", "Experiment entry exists", len(exp_entries) == 1)

    if not exp_entries:
        results.record("Experiment", "Arena matches expected", False, "No experiment")
        results.record("Experiment", "DevicesSchema linked", False, "No experiment")
        results.record("Experiment", "Directory linked", False, "No experiment")
        return

    exp = exp_entries[0]
    results.record("Experiment", "Arena matches expected",
                    exp.get("arena_name") == ARENA_NAME,
                    f"got '{exp.get('arena_name')}', expected '{ARENA_NAME}'")

    schema_entries = (
        acquisition.Experiment.DevicesSchema & {"experiment_name": EXPERIMENT_NAME}
    ).to_dicts()
    has_schema = len(schema_entries) >= 1
    results.record("Experiment", "DevicesSchema linked", has_schema,
                    schema_entries[0]["devices_schema_name"] if has_schema else "missing")

    dir_entries = (
        acquisition.Experiment.Directory & {"experiment_name": EXPERIMENT_NAME}
    ).to_dicts()
    has_dir = len(dir_entries) >= 1
    results.record("Experiment", "Directory linked", has_dir,
                    f"{len(dir_entries)} directories" if has_dir else "missing")


def check_epochs_chunks(results):
    print(f"\n--- Epochs & Chunks ---\n")

    from aeon.dj_pipeline import acquisition

    epochs = (acquisition.Epoch & {"experiment_name": EXPERIMENT_NAME}).to_dicts()
    results.record("Ingestion", "Epochs ingested", len(epochs) >= 1,
                    f"{len(epochs)} epochs")

    chunks = (acquisition.Chunk & {"experiment_name": EXPERIMENT_NAME}).to_dicts()
    results.record("Ingestion", "Chunks ingested", len(chunks) >= 1,
                    f"{len(chunks)} chunks")


def check_epoch_config(results):
    print(f"\n--- EpochConfig ---\n")

    from aeon.dj_pipeline import acquisition, streams

    configs = (acquisition.EpochConfig & {"experiment_name": EXPERIMENT_NAME}).to_dicts()
    results.record("EpochConfig", "EpochConfig populated", len(configs) >= 1,
                    f"{len(configs)} entries")

    if not configs:
        results.record("EpochConfig", "EpochConfig.Meta populated", False, "No EpochConfig")
        results.record("EpochConfig", "Metadata is valid", False, "No EpochConfig")
        results.record("EpochConfig", "No populate errors", False, "No EpochConfig")
        results.record("EpochConfig", "Devices registered", False, "No EpochConfig")
        return

    meta_entries = (
        acquisition.EpochConfig.Meta & {"experiment_name": EXPERIMENT_NAME}
    ).to_dicts()

    if not meta_entries:
        results.record("EpochConfig", "EpochConfig.Meta populated", False, "No Meta entries")
        results.record("EpochConfig", "Metadata is valid", False, "No Meta entries")
    else:
        results.record("EpochConfig", "EpochConfig.Meta populated", True,
                        f"{len(meta_entries)} entries")

        m = meta_entries[0]
        metadata = m["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        is_valid = isinstance(metadata, dict) and len(metadata) > 0
        results.record("EpochConfig", "Metadata is valid", is_valid,
                        f"keys: {list(metadata.keys())}" if is_valid else "empty or not dict")

        if is_valid:
            cameras = metadata.get("cameras", {})
            feeders = metadata.get("feeders", {})
            n_cameras = len(cameras) if isinstance(cameras, dict) else 0
            n_feeders = len(feeders) if isinstance(feeders, dict) else 0
            print(f"    Metadata: {n_cameras} cameras, {n_feeders} feeders")

    try:
        import datajoint as dj
        schema_name = acquisition.EpochConfig.database
        jobs_table = dj.Schema(schema_name).jobs
        if len(jobs_table) == 0:
            results.record("EpochConfig", "No populate errors", True, "jobs table empty")
        elif (errors := (jobs_table & 'status = "error"' & 'table_name LIKE "%epoch_config%"').to_dicts()):
            results.record("EpochConfig", "No populate errors", False, f"{len(errors)} errors")
        else:
            results.record("EpochConfig", "No populate errors", True)
    except Exception as e:
        results.record("EpochConfig", "No populate errors", False, f"Could not check: {e}")

    try:
        devices = streams.DeviceName.to_dicts()
        results.record("EpochConfig", "Devices registered", len(devices) > 0,
                        f"{len(devices)} devices")
    except Exception as e:
        results.record("EpochConfig", "Devices registered", False, f"Error: {e}")


def check_stream_data(results):
    print(f"\n--- Stream Data (populating with limit={STREAM_POPULATE_LIMIT}) ---\n")

    import datajoint as dj
    from aeon.dj_pipeline import acquisition, streams

    stream_tables = []
    for name in dir(streams):
        if name.startswith("_"):
            continue
        obj = getattr(streams, name, None)
        if isinstance(obj, type) and issubclass(obj, dj.Imported) and obj is not dj.Imported:
            stream_tables.append((name, obj))

    if not stream_tables:
        results.record("StreamData", "Stream tables found", False, "No Imported tables")
        return

    results.record("StreamData", "Stream tables found", True,
                    f"{len(stream_tables)} tables")

    populated_results = {}
    error_results = {}
    for table_name, table in stream_tables:
        try:
            table.populate(
                max_calls=STREAM_POPULATE_LIMIT,
                display_progress=False,
                suppress_errors=SUPPRESS_ERRORS,
            )
            count = len(table & {"experiment_name": EXPERIMENT_NAME})
            populated_results[table_name] = count
        except Exception as e:
            error_results[table_name] = str(e)

    populated = {k: v for k, v in populated_results.items() if v > 0}
    zero_count = {k: v for k, v in populated_results.items() if v == 0}

    results.record("StreamData", "Some streams populated", len(populated) > 0,
                    f"{len(populated)}/{len(stream_tables)} tables have data")

    if populated:
        print("\n    Populated tables:")
        for table_name, count in sorted(populated.items()):
            print(f"      ✓ {table_name}: {count} entries")

    if zero_count:
        print("\n    Tables with 0 entries (may be expected for some):")
        for table_name in sorted(zero_count):
            print(f"      – {table_name}: 0 entries")

    if error_results:
        print(f"\n    Tables that raised errors ({len(error_results)}):")
        for table_name, err in sorted(error_results.items()):
            print(f"      ✗ {table_name}: {err}")
        results.record("StreamData", "No populate errors", False,
                        f"{len(error_results)} tables had errors")
    else:
        results.record("StreamData", "No populate errors", True)

    video_populated = {k: v for k, v in populated.items() if "Video" in k}
    results.record("StreamData", "Video streams have data", len(video_populated) > 0,
                    f"{list(video_populated.keys())}" if video_populated else "none")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Validate the foragingOVC pipeline setup.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also populate and check stream data (slow)",
    )
    args = parser.parse_args()

    verify_config_or_exit()

    print("=" * 60)
    print("  foragingOVC Pipeline Validation")
    print(f"  Experiment: {EXPERIMENT_NAME}")
    print(f"  Mode: {'Full (including stream data)' if args.full else 'Quick (structure only)'}")
    print("=" * 60)

    results = CheckResult()

    try:
        check_catalog(results)
        check_tables(results)
        check_experiment(results)
        check_epochs_chunks(results)
        check_epoch_config(results)

        if args.full:
            check_stream_data(results)

    except Exception as e:
        print(f"\n  ✗ Validation failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    all_passed = results.summary()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
