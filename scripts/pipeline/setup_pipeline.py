"""Setup script for the foragingOVC pipeline (v0.4.0, DJ 2.x).

Initializes the DataJoint pipeline following the "Three Decoupled Steps"
architecture from tn/dj2x-upgrade:

  Step 1: Catalog population (StreamType, DeviceType from Pydantic)
  Step 2: Table creation (ExperimentDevice, DeviceDataStream tables)
  Step 3: Data population (EpochConfig.populate(), stream table .populate())

Prerequisites:
  - Credentials configured (see docs/SETUP_GUIDE.md section 3)
  - Dependencies installed (uv sync)
  - Data available on ceph

Usage:
  python scripts/pipeline/setup_pipeline.py              # Run all steps
  python scripts/pipeline/setup_pipeline.py --step 1     # Run single step by number
  python scripts/pipeline/setup_pipeline.py --dry-run    # Show what would happen
"""

import argparse
import json
import sys

# ---------------------------------------------------------------------------
# Experiment configuration — UPDATE THESE FOR YOUR EXPERIMENT
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "ProjectAeonOVC-2026-09-14_001"
DEVICES_SCHEMA = "swc.aeon_exp.foragingOVC.experiment:Experiment"
ARENA_NAME = "circle-2m"  # Pre-populated in lab.Arena Lookup table

# Relative path from the ceph_aeon repository root to the raw data directory.
# e.g. "aeon/data/raw/AEON3/abcBehav0" for foragingABC
DATA_DIR_RELATIVE = "aeon/data/raw/AEON3/experiment_beh/aeon-ovc-beh-2026-09-14"

# Set to True once pipeline is working end-to-end, to allow partial results
# when some entries fail. When False, populate() raises on the first error.
SUPPRESS_ERRORS = False


# ---------------------------------------------------------------------------
# Safety: verify DB config before ANY pipeline imports
# ---------------------------------------------------------------------------
def verify_config_or_exit():
    """Verify the database config BEFORE importing any pipeline modules.

    DJ 2.x auto-loads datajoint.json from the working directory on import.
    The prefix must be set before any pipeline module imports (they read
    db_prefix at import time).
    """
    import datajoint as dj

    if not dj.config.database.database_prefix:
        print(f"\n  ✗ SAFETY CHECK FAILED: no database prefix configured")
        print(f"    Set database_prefix in datajoint.json")
        sys.exit(1)

    # Verify we're not connecting to production DB
    host = dj.config.database.host or ""
    if "aeon-db2" in host:
        print(f"\n  ✗ SAFETY CHECK FAILED: connecting to production host '{host}'")
        print(f"    This script should only run against aeon-db (test), not aeon-db2 (production).")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def print_header(step_num, title):
    print(f"\n{'='*60}")
    print(f"  Step {step_num}: {title}")
    print(f"{'='*60}\n")


def print_ok(msg):
    print(f"  ✓ {msg}")


def print_fail(msg):
    print(f"  ✗ {msg}")


def print_info(msg):
    print(f"  → {msg}")


# ---------------------------------------------------------------------------
# Step 1: Verify configuration
# ---------------------------------------------------------------------------
def step_verify_config(dry_run=False):
    print_header(1, "Verify Configuration")

    if dry_run:
        print_info("Would check: DJ config loaded, DB connection, ceph path")
        return True

    import datajoint as dj

    # Check prefix
    prefix = dj.config.database.database_prefix
    if not prefix:
        print_fail("No database prefix configured in datajoint.json")
        return False
    print_ok(f"Database prefix: {prefix}")

    # Check DB connection
    try:
        conn = dj.conn()
        print_ok(f"Connected to {dj.config.database.host}:{dj.config.database.port}")
    except Exception as e:
        print_fail(f"Cannot connect to database: {e}")
        return False

    # Check repository path
    from aeon.dj_pipeline.utils.paths import get_repository_path

    try:
        repo_path = get_repository_path("ceph_aeon")
        print_ok(f"Repository path: {repo_path}")
    except Exception as e:
        print_fail(f"Cannot resolve repository path: {e}")
        return False

    if not repo_path.exists():
        print_fail(f"Repository path does not exist: {repo_path}")
        return False
    print_ok("Repository path exists on filesystem")

    # Check expected data directory
    if DATA_DIR_RELATIVE.startswith("TODO"):
        print_info("DATA_DIR_RELATIVE not yet configured — skipping data path check")
        return True

    data_dir = repo_path / DATA_DIR_RELATIVE
    if not data_dir.exists():
        print_fail(f"Data directory not found: {data_dir}")
        print_info(f"Expected: {repo_path / DATA_DIR_RELATIVE}")
        return False
    print_ok(f"Data directory exists: {data_dir}")

    # List epoch directories
    epoch_dirs = sorted([f.name for f in data_dir.iterdir() if f.is_dir()])
    print_ok(f"Found {len(epoch_dirs)} epoch directories: {epoch_dirs}")

    return True


# ---------------------------------------------------------------------------
# Step 2: Create base catalog tables
# ---------------------------------------------------------------------------
def step_create_base_tables(dry_run=False):
    print_header(2, "Create Base Catalog Tables (StreamType, DeviceType, DeviceName)")

    if dry_run:
        print_info("Would call: streams_maker.main(create_tables=False)")
        return True

    from aeon.dj_pipeline.utils import streams_maker

    print_info("Running streams_maker.main(create_tables=False)...")
    streams_maker.main(create_tables=False)

    # aeon_mecha >=0.4: the device catalog table is DeviceName (Device was removed)
    from aeon.dj_pipeline.streams import DeviceType, StreamType, DeviceName

    for table_cls, name in [(StreamType, "StreamType"), (DeviceType, "DeviceType"), (DeviceName, "DeviceName")]:
        try:
            len(table_cls())
            print_ok(f"{name} table exists in database")
        except Exception as e:
            print_fail(f"{name} table query failed: {e}")
            return False

    return True


# ---------------------------------------------------------------------------
# Step 3: Populate catalog from Pydantic
# ---------------------------------------------------------------------------
def step_populate_catalog(dry_run=False):
    print_header(3, "Populate Catalog from Pydantic Class")

    if dry_run:
        print_info(f"Would load: {DEVICES_SCHEMA}")
        print_info("Would call: populate_catalog_from_pydantic()")
        return True

    from aeon.dj_pipeline.utils.load_metadata import (
        get_experiment_pydantic,
        populate_catalog_from_pydantic,
    )

    print_info(f"Loading Pydantic class: {DEVICES_SCHEMA}")
    exp_class = get_experiment_pydantic(DEVICES_SCHEMA)
    print_ok(f"Loaded: {exp_class}")

    print_info("Populating StreamType and DeviceType catalog...")
    populate_catalog_from_pydantic(exp_class)

    from aeon.dj_pipeline.streams import DeviceType, StreamType

    n_stream_types = len(StreamType())
    n_device_types = len(DeviceType())

    if n_stream_types == 0:
        print_fail("StreamType has 0 entries")
        return False
    print_ok(f"StreamType entries: {n_stream_types}")

    if n_device_types == 0:
        print_fail("DeviceType has 0 entries")
        return False
    print_ok(f"DeviceType entries: {n_device_types}")

    return True


# ---------------------------------------------------------------------------
# Step 4: Create ExperimentDevice and DeviceDataStream tables
# ---------------------------------------------------------------------------
def step_create_stream_tables(dry_run=False):
    print_header(4, "Create ExperimentDevice & DeviceDataStream Tables")

    if dry_run:
        print_info("Would call: streams_maker.main(create_tables=True)")
        return True

    import datajoint as dj
    from aeon.dj_pipeline.utils import streams_maker

    print_info("Running streams_maker.main(create_tables=True)...")
    streams = streams_maker.main(create_tables=True)

    if streams is None:
        print_fail("streams_maker.main() returned None")
        return False

    table_names = [
        name for name in dir(streams)
        if not name.startswith("_")
        and isinstance(getattr(streams, name, None), type)
        and issubclass(getattr(streams, name), dj.Table)
        and getattr(streams, name) is not dj.Table
    ]

    if len(table_names) == 0:
        print_fail("No DataJoint tables found in streams module after creation")
        return False

    print_ok(f"DataJoint tables in streams module: {len(table_names)}")
    for name in sorted(table_names):
        print_info(f"  {name}")

    return True


# ---------------------------------------------------------------------------
# Step 5: Register experiment
# ---------------------------------------------------------------------------
def step_register_experiment(dry_run=False):
    print_header(5, "Register Experiment")

    if EXPERIMENT_NAME.startswith("TODO"):
        print_fail("EXPERIMENT_NAME not yet configured — update it at the top of this script")
        return False

    if dry_run:
        print_info(f"Would register experiment: {EXPERIMENT_NAME}")
        print_info(f"  devices_schema: {DEVICES_SCHEMA}")
        print_info(f"  arena: {ARENA_NAME}")
        return True

    from aeon.dj_pipeline import acquisition
    from aeon.dj_pipeline.utils.create_experiment import create_experiment

    print_info(f"Registering DevicesSchema: {DEVICES_SCHEMA}")
    acquisition.DevicesSchema.insert1(
        {"devices_schema_name": DEVICES_SCHEMA},
        skip_duplicates=True,
    )
    print_ok("DevicesSchema registered")

    if acquisition.Experiment & {"experiment_name": EXPERIMENT_NAME}:
        # Already registered by the ephys runbook (step01_register_experiment.py),
        # which does not insert the DevicesSchema link — add only that.
        print_info(f"Experiment already exists: {EXPERIMENT_NAME} — linking DevicesSchema only")
        acquisition.Experiment.DevicesSchema.insert1(
            {"experiment_name": EXPERIMENT_NAME, "devices_schema_name": DEVICES_SCHEMA},
            skip_duplicates=True,
        )
    else:
        from aeon.dj_pipeline.utils.paths import get_repository_path

        # Pass raw_dir/location explicitly: create_experiment otherwise parses
        # "<exp>-<machine>" from the name, which breaks on dated names
        # like "ProjectAeonOVC-2026-09-14_001".
        print_info(f"Creating experiment: {EXPERIMENT_NAME}")
        create_experiment(
            experiment_name=EXPERIMENT_NAME,
            experiment_type="foraging",
            devices_schema_name=DEVICES_SCHEMA,
            arena_name=ARENA_NAME,
            location="AEON3",
            raw_dir=get_repository_path("ceph_aeon") / DATA_DIR_RELATIVE,
        )
        print_ok(f"Experiment created: {EXPERIMENT_NAME}")

    exp = (acquisition.Experiment & {"experiment_name": EXPERIMENT_NAME}).fetch1()
    print_ok(f"  start_time: {exp['experiment_start_time']}")
    print_ok(f"  arena: {exp['arena_name']}")
    print_ok(f"  location: {exp['location']}")

    if exp["arena_name"] != ARENA_NAME:
        print_fail(f"  Arena mismatch: got '{exp['arena_name']}', expected '{ARENA_NAME}'")
        return False

    dirs = (acquisition.Experiment.Directory & {"experiment_name": EXPERIMENT_NAME}).to_dicts()
    if not dirs:
        print_fail("  No directories registered for experiment")
        return False
    for d in dirs:
        print_ok(f"  directory: {d['directory_type']} → {d['directory_path']}")

    schema_link = (acquisition.Experiment.DevicesSchema & {"experiment_name": EXPERIMENT_NAME}).fetch1()
    print_ok(f"  devices_schema: {schema_link['devices_schema_name']}")

    return True


# ---------------------------------------------------------------------------
# Step 6: Ingest epochs
# ---------------------------------------------------------------------------
def step_ingest_epochs(dry_run=False):
    print_header(6, "Ingest Epochs")

    if dry_run:
        print_info(f"Would call: Epoch.ingest_epochs('{EXPERIMENT_NAME}')")
        return True

    from aeon.dj_pipeline import acquisition

    print_info(f"Ingesting epochs for {EXPERIMENT_NAME}...")
    acquisition.Epoch.ingest_epochs(EXPERIMENT_NAME)

    epochs = (acquisition.Epoch & {"experiment_name": EXPERIMENT_NAME}).to_dicts()
    if len(epochs) == 0:
        print_fail("No epochs were ingested")
        return False

    print_ok(f"Ingested {len(epochs)} epochs")
    for ep in epochs:
        print_info(f"  {ep['epoch_start']} → {ep.get('epoch_dir', 'N/A')}")

    return True


# ---------------------------------------------------------------------------
# Step 7: Ingest chunks
# ---------------------------------------------------------------------------
def step_ingest_chunks(dry_run=False):
    print_header(7, "Ingest Chunks")

    if dry_run:
        print_info(f"Would call: Chunk.ingest_chunks('{EXPERIMENT_NAME}')")
        return True

    from aeon.dj_pipeline import acquisition

    print_info(f"Ingesting chunks for {EXPERIMENT_NAME}...")
    acquisition.Chunk.ingest_chunks(EXPERIMENT_NAME)

    chunks = (acquisition.Chunk & {"experiment_name": EXPERIMENT_NAME}).to_dicts()
    if len(chunks) == 0:
        print_fail("No chunks were ingested")
        return False

    print_ok(f"Ingested {len(chunks)} chunks")

    return True


# ---------------------------------------------------------------------------
# Step 8: Populate EpochConfig
# ---------------------------------------------------------------------------
def step_populate_epoch_config(dry_run=False):
    print_header(8, "Populate EpochConfig (metadata + device registration)")

    if dry_run:
        print_info("Would call: EpochConfig.populate()")
        return True

    from aeon.dj_pipeline import acquisition

    print_info(f"Running EpochConfig.populate(suppress_errors={SUPPRESS_ERRORS})...")
    acquisition.EpochConfig.populate(
        {"experiment_name": EXPERIMENT_NAME}, display_progress=True, suppress_errors=SUPPRESS_ERRORS
    )

    import datajoint as dj
    schema_name = acquisition.EpochConfig.database
    jobs_table = dj.Schema(schema_name).jobs

    if len(jobs_table) > 0:
        errors = (jobs_table & 'status = "error"' & 'table_name LIKE "%epoch_config%"').to_dicts()
        if errors:
            print_fail(f"EpochConfig.populate() had {len(errors)} errors in jobs table:")
            for err in errors:
                print_info(f"  Key: {err.get('key', 'N/A')} — {err.get('error_message', 'N/A')[:200]}")
            return False

    configs = (acquisition.EpochConfig & {"experiment_name": EXPERIMENT_NAME}).to_dicts()
    if len(configs) == 0:
        print_fail("No EpochConfig entries created")
        return False
    print_ok(f"EpochConfig entries: {len(configs)}")

    meta = (acquisition.EpochConfig.Meta & {"experiment_name": EXPERIMENT_NAME}).to_dicts()
    if not meta:
        print_fail("No EpochConfig.Meta entries")
        return False

    m = meta[0]
    metadata = m["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    if not isinstance(metadata, dict) or len(metadata) == 0:
        print_fail(f"EpochConfig.Meta.metadata is empty or not a dict: {type(metadata)}")
        return False

    print_info(f"Metadata (rig) top-level keys: {list(metadata.keys())}")
    cameras = metadata.get("cameras", {})
    feeders = metadata.get("feeders", {})
    n_cameras = len(cameras) if isinstance(cameras, dict) else 0
    n_feeders = len(feeders) if isinstance(feeders, dict) else 0
    print_ok(f"Metadata loaded: {n_cameras} cameras, {n_feeders} feeders")

    return True


# ---------------------------------------------------------------------------
# Step 9: Populate stream tables
# ---------------------------------------------------------------------------
def step_populate_streams(dry_run=False):
    print_header(9, "Populate Stream Tables (CameraVideo, SoundCard, Environment, ...)")

    if dry_run:
        print_info(f"Would call: .populate() on all Imported streams.* tables for '{EXPERIMENT_NAME}'")
        print_info("Would call: acquisition.LightEvents/MessageLog/EnvironmentState.populate()")
        return True

    import datajoint as dj
    from aeon.dj_pipeline import acquisition, streams

    restriction = {"experiment_name": EXPERIMENT_NAME}
    stream_tables = [
        (name, cls) for name, cls in vars(streams).items()
        if isinstance(cls, type) and issubclass(cls, dj.Imported)
    ]
    env_tables = [
        (f"acquisition.{cls.__name__}", cls)
        for cls in (acquisition.LightEvents, acquisition.MessageLog, acquisition.EnvironmentState)
    ]

    for name, table_cls in sorted(stream_tables) + env_tables:
        print_info(f"Populating {name}...")
        table_cls.populate(restriction, display_progress=True, suppress_errors=SUPPRESS_ERRORS)
        print_ok(f"{name}: {len(table_cls & restriction)} rows")

    # Required for SLEAP tracking (foragingABC pose.SleapRawTracking -> streams.CameraVideo)
    n_chunks = len(acquisition.Chunk & restriction)
    n_top = len(streams.CameraVideo & restriction & {"device_name": "CameraTop"})
    if n_top == 0:
        print_fail("streams.CameraVideo has no CameraTop rows for this experiment")
        return False
    print_ok(f"CameraTop videos: {n_top} / {n_chunks} chunks")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
STEPS = [
    ("verify_config", step_verify_config),
    ("create_base_tables", step_create_base_tables),
    ("populate_catalog", step_populate_catalog),
    ("create_stream_tables", step_create_stream_tables),
    ("register_experiment", step_register_experiment),
    ("ingest_epochs", step_ingest_epochs),
    ("ingest_chunks", step_ingest_chunks),
    ("populate_epoch_config", step_populate_epoch_config),
    ("populate_streams", step_populate_streams),
]


def main():
    parser = argparse.ArgumentParser(
        description="Set up the foragingOVC pipeline on HPC.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  1  verify_config        Check DJ config, DB connection, ceph path
  2  create_base_tables   Create StreamType, DeviceType, DeviceName tables
  3  populate_catalog     Fill catalog from Pydantic Experiment class
  4  create_stream_tables Create ExperimentDevice & DeviceDataStream tables
  5  register_experiment  Insert Experiment, Directory, DevicesSchema
                          (links DevicesSchema only if already registered)
  6  ingest_epochs        Detect and ingest epochs from filesystem
  7  ingest_chunks        Detect and ingest chunks from filesystem
  8  populate_epoch_config Run EpochConfig.populate() for metadata
  9  populate_streams     Populate streams.* (CameraVideo, ...) + Environment tables

Steps 2-4 run once per database; steps 5-9 run once per session.
        """,
    )
    parser.add_argument(
        "--step",
        type=int,
        choices=range(1, len(STEPS) + 1),
        help=f"Run only the specified step (1-{len(STEPS)})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what each step would do without executing",
    )
    args = parser.parse_args()

    if not args.dry_run:
        verify_config_or_exit()

    import datajoint as dj
    print("=" * 60)
    print("  foragingOVC Pipeline Setup")
    print(f"  Experiment: {EXPERIMENT_NAME}")
    print(f"  DB prefix: {dj.config.database.database_prefix}")
    if args.dry_run:
        print("  Mode: DRY RUN")
    print("=" * 60)

    if args.step:
        step_name, step_func = STEPS[args.step - 1]
        try:
            success = step_func(dry_run=args.dry_run)
            if not success:
                print(f"\n✗ Step {args.step} ({step_name}) failed.")
                sys.exit(1)
        except Exception as e:
            print_fail(f"Step {args.step} ({step_name}) raised an exception: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        print(f"\n✓ Step {args.step} ({step_name}) completed.")
    else:
        for i, (step_name, step_func) in enumerate(STEPS, 1):
            try:
                success = step_func(dry_run=args.dry_run)
                if not success:
                    print(f"\n✗ Step {i} ({step_name}) failed. Stopping.")
                    sys.exit(1)
            except Exception as e:
                print_fail(f"Step {i} ({step_name}) raised an exception: {e}")
                import traceback
                traceback.print_exc()
                print(f"\n✗ Pipeline setup failed at step {i}. Fix the issue and re-run with --step {i}")
                sys.exit(1)

        print("\n" + "=" * 60)
        print("  ✓ All steps completed successfully!")
        print("  Next: run scripts/pipeline/validate_pipeline.py to verify")
        print("=" * 60)


if __name__ == "__main__":
    main()
