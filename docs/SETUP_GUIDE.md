# Setup Guide: foragingOVC Pipeline (v0.4.0)

## Goal

Set up the DataJoint pipeline so you can query your OVC experiment data from Python. This uses DataJoint 2.x with the Pydantic-based config and the generic template approach.

## What you'll have after setup

- `aeon_mecha` as a submodule (the core pipeline code)
- Database schemas under `aeon_foragingOVC_` with your experiment data
- The ability to import pipeline tables and fetch data in your analysis scripts

## Prerequisites

- Database credentials for aeon-db (you should have received these by email)
- MySQL write permissions for `aeon_foragingOVC_%` schemas
- Access to the ceph aeon data directory (via HPC)
- Python 3.11 or 3.12 (not 3.13+) and [uv](https://docs.astral.sh/uv/) installed

---

## 1. Check out the branch and initialize the submodule

```bash
cd AEON_OVC
git checkout es/aeon-mecha-setup
git submodule update --init --recursive
```

The submodule at `submodules/aeon_mecha/` tracks the `foragingOVC` branch. If you want your own branch:

```bash
cd submodules/aeon_mecha
git checkout -b <your_branch>
cd ../..
```

---

## 2. Install dependencies

```bash
uv sync
```

Key dependencies:
- `datajoint>=2.1.0` (DJ 2.x)
- `aeon-mecha` from the submodule (editable)
- `swc-aeon-ovc` from `aeon_exp_OVC`

Use `uv run python` (not bare `python`) to run scripts.

---

## 3. Configure the database connection

DJ 2.x uses `datajoint.json` for non-secret config (already in the repo) and a `.secrets/` directory for credentials (gitignored, never committed).

**Check your database prefix** in `datajoint.json`:
```json
{
  "database": {
    "host": "aeon-db",
    "port": 3306,
    "reconnect": true,
    "database_prefix": "aeon_foragingOVC_"
  }
}
```

Update `host` and `port` if you're connecting via SSH tunnel.

**Set up your credentials:**

```bash
mkdir -p .secrets
echo "your_username" > .secrets/database.user
echo "your_password" > .secrets/database.password
```

Scripts can be run from anywhere within the project. DJ 2.x auto-discovers `datajoint.json` and `.secrets/` by searching upward to the nearest `.git` boundary.

---

## 4. Configure your experiment

The experiment configuration is already set at the top of `scripts/pipeline/setup_pipeline.py`:

- `EXPERIMENT_NAME` = `ovc01-aeon3`
- `DATA_DIR_RELATIVE` = `aeon/data/raw/AEON3/ovc01`

The same experiment name is set in `scripts/pipeline/validate_pipeline.py`. If these ever need to change, update both files.

---

## 5. Set up the pipeline and test

Run all steps:
```bash
uv run python scripts/pipeline/setup_pipeline.py
```

Or step by step:
```bash
uv run python scripts/pipeline/setup_pipeline.py --step 1  # Verify config
uv run python scripts/pipeline/setup_pipeline.py --step 2  # Create base tables
# ... etc (steps 1-8)
```

Preview without executing:
```bash
uv run python scripts/pipeline/setup_pipeline.py --dry-run
```

Validate the pipeline:
```bash
uv run python scripts/pipeline/validate_pipeline.py          # Quick mode
uv run python scripts/pipeline/validate_pipeline.py --full   # Full mode (includes stream data)
```

---

## 6. Using the pipeline in your analysis

Once the pipeline is set up, you can import and query data from any Python script:

```python
import datajoint as dj

# These imports create the connection using datajoint.json + .secrets/
from aeon.dj_pipeline import acquisition, streams

# Query your experiment's chunks
chunks = (acquisition.Chunk & {"experiment_name": "ovc01-aeon3"}).to_pandas()

# Fetch epoch metadata
configs = (acquisition.EpochConfig & {"experiment_name": "ovc01-aeon3"}).to_dicts()
```

---

## Commit streams.py

After running setup step 4, `streams_maker.py` auto-generates `aeon/dj_pipeline/streams.py` inside the submodule. This file is experiment-specific. It should be committed to your aeon_mecha branch.

```bash
cd submodules/aeon_mecha
git add aeon/dj_pipeline/streams.py
git commit -m "Add auto-generated streams.py for foragingOVC"
git push origin <your_branch>

cd ../..
git add submodules/aeon_mecha
git commit -m "Update aeon_mecha submodule"
git push
```

---

## Changing your data path

If your data moves or you need to point to a different location, update two things:

1. `DATA_DIR_RELATIVE` in `scripts/pipeline/setup_pipeline.py`
2. The `directory_path` in the `Experiment.Directory` table (this is set during step 5 and stored in the DB)

To update the directory in the DB, you can either drop your schemas and re-run setup, or update the entry directly.

---

## Starting over (dropping schemas)

If you need to start fresh:

```bash
uv run python -c "
import datajoint as dj
prefix = dj.config.database.database_prefix
schemas = [s for s in dj.list_schemas() if s.startswith(prefix)]
print(f'Found {len(schemas)} schemas with prefix \"{prefix}\":')
for s in schemas:
    print(f'  {s}')
confirm = input('Drop all? (yes/no): ')
if confirm == 'yes':
    while schemas:
        remaining = []
        for s in schemas:
            try:
                dj.Schema(s).drop()
                print(f'  Dropped {s}')
            except Exception:
                remaining.append(s)
        if len(remaining) == len(schemas):
            print(f'ERROR: Cannot drop remaining schemas: {remaining}')
            break
        schemas = remaining
    print('Done.')
"
```

---

## HPC-specific notes

- **SSH:** `ssh -J <your_username>@ssh.swc.ucl.ac.uk <your_username>@hpc-gw2`
- **uv:** Load the uv module first: `module load uv`
- **Returning to work:** Run `module load uv` each time you SSH in before using `uv run` or `uv sync`.
- **ceph path:** `/ceph/aeon`
