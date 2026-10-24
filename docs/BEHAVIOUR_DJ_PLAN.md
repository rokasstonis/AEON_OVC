# Behaviour in DataJoint: plan and progress

> Copied from the Claude Code plan file (`~/.claude/plans/in-addition-to-ephys-silly-coral.md`) on 2026-10-21 so it can be continued from another PC.
> To resume in a new Claude session, say: "Read docs/BEHAVIOUR_DJ_PLAN.md and continue from 'Next up'".

## Progress (as of 2026-10-21)

**Done**
- Fixed `Device` → `DeviceName` in `scripts/pipeline/setup_pipeline.py` and `validate_pipeline.py` (aeon_mecha ≥0.4 renamed the table).
- Ran `setup_pipeline.py --step 2`, `--step 3` and `--step 4` only. These are once per database. Don't run the full script: steps 1–4 don't need re-running. Since 2026-10-22, `--step 5` is safe for sessions that step01 already registered (it only adds the DevicesSchema link).
- Added `@data_reader sound_events` (Harp register 32, column `sound_index`) to `SoundCard` in the rig repo `aeon_exp_OVC`, pushed as commit `50ab39f`. Bumped it in `uv.lock`.
- Fixed aeon_mecha `load_metadata.get_device_class_from_field` to discover optional rig fields (`SoundCard | None`). Committed on aeon_mecha branch `my-changes` (`aa0d5e51`) and pushed. Worth an upstream PR.
- Regenerated `aeon_mecha/aeon/dj_pipeline/streams.py` (`SoundCardSoundEvents` and the others), committed as `148e3d67` and pushed.
- Main repo: committed the script fixes, uv.lock and the submodule pointer. Added the foragingABC_analysis submodule.
- Caveat: `.gitmodules` says aeon_mecha tracks branch `foragingOVC`, but the commits are on `my-changes`. Don't run `git submodule update --remote` until that is reconciled.

- (2026-10-22) Instead of a separate `step01b_behaviour_ingest.py`, `setup_pipeline.py` steps 5–9 are now the per-session behaviour ingest:
  - Step 5 only links DevicesSchema if step01 already registered the experiment. Before this, `create_experiment` mis-parsed dated names like `ProjectAeonOVC-2026-09-14_001` as `<exp>-<machine>` and looked for `raw/14_001/...`.
  - Step 8 is restricted to `EXPERIMENT_NAME`.
  - The new step 9 populates every `streams.*` Imported table plus `acquisition.LightEvents`, `MessageLog` and `EnvironmentState`.

**Next up**
1. Per-session behaviour ingestion. Set `EXPERIMENT_NAME` and `DATA_DIR_RELATIVE` in `setup_pipeline.py` to the session, then run `--step 5`, `7`, `8` and `9` after the ephys step01. Steps 2–4 are once per database and already done. Done for `ProjectAeonOVC-2026-09-14_001` once step 9 passes.
2. Install foragingABC as a uv path dependency (see Step 2). Then run `import foragingabc.dj_pipeline.pose` as a smoke test.
3. Write `AEON_OVC_beh/dj_integration/compat.py`, which makes foragingABC call the legacy `sleap-track`. Insert the `SleapTrackingMethod` and `CameraView` rows, then test one chunk on a GPU node.

**Standing decisions / preferences**
- SLEAP inference must use the user's legacy SLEAP 1.4 `sleap_legacy` TensorFlow env. The models perform poorly under sleap-nn (SLEAP ≥1.5).
- `scripts/hpc/submit_sleap.sh` belongs to a separate behaviour workflow. Don't modify it, and don't use it for DataJoint.
- The user has write access to `/ceph/aeon/aeon/data/processed`.
- Shelter and object positions move within a session and are registered separately. `ArenaLayout` is a TODO placeholder for now.
- CameraTop and the arena match foragingABC's AEON3 setup, so its calibration is reused.
- Ephys and behaviour share the per-session experiment names (`ProjectAeonOVC-<date>_NNN`).

---

# Plan: SLEAP tracking + OVC behaviour variables in DataJoint, alongside ephys

## Context
Ephys is already queryable through DataJoint (`aeon_foragingOVC_*` schemas on `aeondj`, spike times on the HARP clock via `SyncedSpikes` / `UnitMatching.Spikes`). Behaviour is not. The goal is to query the following in the same DataJoint style:
- position, velocity and heading from SLEAP, reusing SWC foragingABC_analysis
- sound-card stimulus events, which trigger the escapes
- dark periods, when Environment LightEvents channels 3 and 7 are 0

Then combine them into escape trials labelled light or dark and aligned with spikes.

### Key facts from exploration
- **foragingABC_analysis** (local copy at `/media/rokas/.../foragingABC_analysis`, HEAD `5a25dde`) is a DataJoint 2.x package built on aeon_mecha.
  - It names schemas with `get_schema_name(...)`, so the `aeon_foragingOVC_` prefix from your `datajoint.json` applies automatically.
  - `foragingabc/dj_pipeline/pose.py` holds the generic SLEAP chain:
    - `SleapTrackingMethod` (+ `CameraView`): Manual.
    - `SleapRawTracking`: `-> streams.CameraVideo`. It runs `sleap track` itself and stores HARP-timed movement xarray as `<xarray@dj_store>`.
    - `PoseQC`, `PositionPixel`, `PositionCM` (undistort + homography from `/ceph/aeon/aeon/sleap_tracking/cm_conversion/camera_calibration.npz`, which covers AEON3 CameraTop) and `HeadingDirection`.
  - Speed comes from `utils/behaviour.py:compute_velocity` (gradient, Gaussian σ=5 frames, cm/s), with chunk-edge padding in `get_extended_position`.
  - `behav.AllMovementBouts` is generic.
  - `abc_task.py`, `feeder.py` and `behav.DiggingBoutMovement` are specific to foraging ABC. **Do not import them.**
  - Its custom-event pattern (`abc_task.StateTransitions`): an Imported table keyed `-> acquisition.Chunk` that loads a raw stream with a `swc.aeon.io.reader` subclass via `aeon_api.load`. That is the template for sound-card and light tables.
- **Your SLEAP model skeleton matches foragingABC exactly** (`sleap/model/260921_*`): nose, head, left/right_ear, spine1–4. So `position.py` and `heading_direction.py` work unchanged.
- **Version gap.** foragingABC pins aeon_mecha 0.4.7 (branch `ar/core`). Your submodule is at `ff1ce425` (`foragingOVC`, ~0.4.3). Both use DataJoint ≥2.3.2 and the `xarray` codec exists in yours (`aeon/dj_pipeline/utils/codec.py:226`), so compatibility is likely but must be verified.
- **Raw data** (`.../AEON3/experiment_beh/aeon-ovc-beh-2026-09-28/2026-09-28T155241Z/`):
  - `SoundCard_32_*.bin`: Harp event reg 32, one U16 = soundIndex played (28 events).
  - `SoundCard_37_*.bin`: the play command `[index, attL, attR]`, ~0.6 ms earlier.
  - `SoundCard_8_*.bin`: heartbeat. Ignore it.
  - `Metadata.json` `soundBank` maps index → wav name (e.g. 14 = MAX_10kHz_tone_9s, 20 = MAX_17kHz_to_20kHz_3x3s).
  - `Environment_LightEvents_*.csv` (`Seconds, Value.Channel, Value.Value`) logs each channel every ~60 s. Channels 3 and 7 are identical: 78 by day, 219 at dusk and dawn, 253 at night.
  - Dark episodes appear as **one `0` sample followed by a logging gap** (2–37 min) until 78 resumes. They match the "Dark"/"Light" annotations in `Environment_MessageLog`.
- **aeon_mecha in this repo:**
  - `acquisition.LightEvents`, `MessageLog` and `EnvironmentState` already exist (`acquisition.py:572-613`, `_make_environment_stream` at `:724`).
  - The OVC rig package (`swc/aeon_exp/foragingOVC/rig.py:294-334`) has `Environment.light_events` but no SoundCard reader.
  - The OVC `Camera` has a `video` reader, so `streams_maker` will generate `streams.CameraVideo`. That is exactly what foragingABC's `SleapRawTracking` hangs off.
- **Gap blocking all behaviour tables:**
  - The ephys runbook `step01_register_experiment.py` never inserts `Experiment.DevicesSchema`, never runs `Chunk.ingest_chunks`, `EpochConfig.populate` or the stream tables. The committed `streams.py` only has the catalogs.
  - `scripts/pipeline/setup_pipeline.py` already does these steps, but for a different experiment name (`ovc01-aeon3`).

## Approach
1. Use foragingABC as a library: a git submodule, installed editable with `--no-deps` so your aeon_mecha stays the one in use. Activate only its `pose` schema and generic utils.
2. Add a small OVC-specific schema `aeon_foragingOVC_ovc_behav` for sound, dark and trial tables, in `AEON_OVC_beh/dj_integration/`, which is currently empty.
3. Upstream nothing to foragingABC. Adapt by configuration, not by forking.

### Step 0: Fix stale `Device` references (blocks setup step 2)
Running `setup_pipeline.py` failed with `ImportError: cannot import name 'Device' from 'aeon.dj_pipeline.streams'`. Since aeon_mecha v0.4 (`streams_maker.py:72,86,292`), the catalog is `StreamType` / `DeviceType` / **`DeviceName`**, and there is no `Device` table. `streams_maker.main(create_tables=False)` itself ran fine; only the script's post-check uses the old name.
- `scripts/pipeline/setup_pipeline.py`:
  - :163 → `from aeon.dj_pipeline.streams import DeviceType, StreamType, DeviceName`
  - :165 → `(DeviceName, "DeviceName")`
  - Update the header text at :152 and :443.
- `scripts/pipeline/validate_pipeline.py:242`: `streams.Device.to_dicts()` → `streams.DeviceName.to_dicts()`. Check the surrounding lines for `Device`-specific columns.
- Then run **only** `python scripts/pipeline/setup_pipeline.py --step 2`, `--step 3` and `--step 4`, which are once per database. **Never run the full script or `--step 5`**: step 5 would register an extra `ovc01-aeon3` experiment. Steps 6–8 for real sessions are handled by step01b below.
- After `--step 4`, commit the regenerated `aeon_mecha/aeon/dj_pipeline/streams.py`, which should now contain `CameraVideo`, `EnvironmentLightEvents` and the other stream tables, so every machine imports the same tables.

### Step 1: Make the behaviour side ingestible (don't re-run all of `setup_pipeline.py`)
`step01_register_experiment.py` already inserts `Experiment`, `Experiment.Directory` (raw and raw-ephys), `Experiment.Subject` and the behaviour `acquisition.Epoch` (via `Epoch.ingest_epochs`). Its `ingest_chunks` only creates **`EphysChunk`**, not behaviour `acquisition.Chunk`. Compared with `scripts/pipeline/setup_pipeline.py`:

| setup_pipeline step | Needed? | Why |
|---|---|---|
| 2 `streams_maker.main(create_tables=False)` | **Once per database** | Base device and stream catalog tables |
| 3 `populate_catalog_from_pydantic(foragingOVC Experiment)` | **Once per database** | Fills DeviceType and StreamType from the OVC rig |
| 4 `streams_maker.main(create_tables=True)` | **Once per database** | Generates `streams.CameraVideo` and the other stream tables. The committed `streams.py` still has only the catalog classes, so this has not been done |
| 5 `create_experiment(...)` | **No** | step01 already registered the experiment. Running it would make a second `ovc01-aeon3` experiment |
| 5 (part) DevicesSchema link | **Yes, per experiment** | `acquisition.DevicesSchema.insert1({"devices_schema_name": "swc.aeon_exp.foragingOVC.experiment:Experiment"}, skip_duplicates=True)` + `acquisition.Experiment.DevicesSchema.insert1({experiment_name, devices_schema_name})`. step01 never sets it, and readers can't be resolved without it |
| 6 `Epoch.ingest_epochs` | No | step01 already does it |
| 7 `Chunk.ingest_chunks` | **Yes, per experiment** | Behaviour hourly chunks, the key for all pose and behaviour tables |
| 8 `EpochConfig.populate` | **Yes, per experiment** | Rig metadata (sound bank, devices) per epoch |

Implementation:
- A new `step01b_behaviour_ingest.py` next to the ephys runbook, using the same `EXPERIMENT_NAME` constant as step01. It does the following:
  - `ensure_stream_tables()`: steps 2–4. It is skipped if `streams.CameraVideo` already exists, and it imports the step functions from `setup_pipeline.py`.
  - `link_devices_schema(exp)`.
  - `acquisition.Chunk.ingest_chunks(exp)`.
  - `EpochConfig.populate(exp)`.
  - Populate the `streams.*` tables for the experiment, including `CameraVideo`.
  - Populate `acquisition.LightEvents`, `MessageLog` and `EnvironmentState`.
- It is idempotent: it uses `skip_duplicates` and populate restrictions, and it can be re-run for each new session.
- Optionally, later: fold `link_devices_schema` and `Chunk.ingest_chunks` into step01 itself, so new sessions are behaviour-ready in one pass.

### Step 2: Bring in foragingABC
- The submodule is already added at `submodules/foragingABC_analysis`. Add it as a uv path dependency, not `uv pip install --no-deps`, because the next `uv sync` would remove that. In `pyproject.toml`:
  - dependencies: `foragingabc-analysis`, `movement`, `h5netcdf`
  - sources: `foragingabc-analysis = { path = "./submodules/foragingABC_analysis", editable = true }`
  - `override-dependencies`: add `swc-aeon-rigs-foragingabc; sys_platform == 'never'`. This drops the private foragingABC rig package, which isn't needed.
- Add `movement==0.15.0`, `opencv-python` and `netCDF4`/`h5netcdf` to `pyproject.toml` / `environment.yml`.
- Smoke test: `import foragingabc.dj_pipeline.pose` with your `datajoint.json`. It should create `aeon_foragingOVC_pose`.
  - Fix any API drift between aeon_mecha 0.4.3 and 0.4.7 inside a thin compatibility shim in `AEON_OVC_beh/dj_integration/compat.py`, not in foragingABC.
  - Things to check: `get_stream_reader_for_epoch` signature, the `swc.aeon.io.api` import name, and `Experiment.get_data_directory`.
- Configure through inserts:
  - Insert a `SleapTrackingMethod` row: `sleap_method="ovc_top_260921_legacy"`, `multi_animal=0`, model paths `/ceph/branco/Rokas/Ephys/SLEAP/210926/260921_*`. The paths are recorded for provenance; inference itself happens in your own environment.
  - Insert a `CameraView` row for `CameraTop`.
- Calibration: CameraTop and the arena are the same in both experiments (confirmed), so use foragingABC's `camera_calibration.npz` unchanged. Keep one overlay of `PositionCM.canvas_position()` as a sanity check only.

### Step 2b: Legacy SLEAP 1.4 inference inside DataJoint populate
Your models perform poorly under sleap-nn, so inference must use your `sleap_legacy` TensorFlow env. Everything still runs through `SleapRawTracking.populate()`. `scripts/hpc/submit_sleap.sh` is left untouched: it belongs to the separate behaviour workflow.

**How it works.** `pipeline.run_sleap_pipeline` (`foragingabc/dj_pipeline/utils/sleap/pipeline.py`) does `from .track import run_inference` and calls it through the module-level name. `AEON_OVC_beh/dj_integration/compat.py` replaces `foragingabc.dj_pipeline.utils.sleap.pipeline.run_inference` with `run_inference_legacy`, which has the same signature:
- It calls `${SLEAP_LEGACY_BIN:-/ceph/apps/ubuntu-20/packages/miniconda/23.10.0/envs/sleap_legacy/bin}/sleap-track <video> --model <centroid> --model <centered> --output <output_slp> --batch-size 8 --max_tracks 1 --max_instances 1 --peak-threshold 0.2 --verbosity json`. These are the flags from your legacy script, via `subprocess`, writing to a log file.
- It writes to a temporary file, then renames it to `output_slp`, and checks that the file is non-empty.
- It sets `TF_FORCE_GPU_ALLOW_GROWTH=true`, so several processes can share one GPU.

Everything else in foragingABC stays as it is:
- the pre-checks (video, model folders, GPU, writable output dir)
- `.slp` → movement xarray conversion and HARP alignment
- post-checks
- `keep_slp=True`, which keeps the file at `/ceph/aeon/aeon/data/processed/sleap_tracking/ovc_top_260921_legacy/<...>/CameraTop/SleapKeypoints_<chunk_start>.slp`. You have write access there.

`compat.apply()` is idempotent and also logs which SLEAP binary runs. Every OVC entry point imports it before calling populate.

- **Populate driver** `AEON_OVC_beh/dj_integration/populate_behaviour.py` (CLI `--experiment`, `--stage pose|behav|all`, `--processes`, `--max-calls`):
  - `compat.apply()`, then `SleapRawTracking.populate(exp, reserve_jobs=True)`.
  - Then `PoseQC` → `PositionCM` / `HeadingDirection` → `Speed`, then the OVC behaviour tables (Step 3).
- **SLURM** `scripts/hpc/submit_pose.sh`, adapted from foragingABC `slurm/run_pose_per_experiment.sh`. It is an array of workers, each running the driver for one experiment, and DataJoint job reservation splits the chunks.
  - Activate the main `.venv` (DataJoint, movement), not the legacy env. The legacy env is only called as a subprocess.
  - Partition `gpu_branco`, 1 GPU and 4 CPUs per task. Start with `PROCESSES=1`, and raise it once GPU memory sharing is confirmed.
- **First-hour checks:**
  - **Legacy env on the node.** The `ubuntu-20` miniconda env must run on the GPU nodes. If it doesn't, point `SLEAP_LEGACY_BIN` to wherever `sleap_legacy` works.
  - **Frame count.** Legacy `.slp` files can omit trailing frames that have no instance. If `convert_time_to_harp_time` fails the length check, wrap `convert_slp_to_xarr` in `compat.py` to pad the frames up to the video frame count with NaN.
  - Pre-checks need `sleap_io` importable in the main env. It comes with `movement`.

### Step 3: OVC behaviour schema (`AEON_OVC_beh/dj_integration/ovc_behav.py`)
All tables use the foragingABC `abc_task.StateTransitions` pattern: `get_stream_reader_for_epoch` or a custom reader → `io_api.load(root=Experiment.get_data_directories(key), start=chunk_start, end=chunk_end)`.

- **`SoundStimulus`** (Imported, `-> acquisition.Experiment, sound_index`): parse `Metadata.json` `soundBank`, the same way `abc_task.StateToRuleMap` parses metaStates. Fields: `sound_name`, `freq_khz`, `pattern` (tone/pulses/sweep), `duration_s`, `is_max`.
- **Preferred, confirmed missing after setup step 4 on 2026-10-21:** add a `@data_reader sound_events` to `SoundCard` in the rig repo `aeon_exp_OVC` (`swc/aeon_exp/foragingOVC/rig.py`), returning `reader.Harp(f"{pattern}_32_*", columns=["sound_index"])`. Reinstall it, then re-run setup steps 3 and 4. That auto-generates `streams.SoundCardSoundEvents`, and `SoundEvents` below becomes unnecessary.
- **`SoundEvents`** (fallback if the rig repo can't change; Imported, `-> acquisition.Chunk`): `timestamps <blob>`, `sound_index <blob>`, `sample_count`.
  - Reader: `class SoundEvent(swc.aeon.io.reader.Harp)` with pattern `SoundCard_32_*` and columns `["sound_index"]`, defined in `AEON_OVC_beh/dj_integration/readers.py`.
  - Later, optionally upstream it as a `@data_reader` on `SoundCard` in the swc-aeon-ovc rig package. `streams_maker` would then auto-generate a stream table.
- **`DarkParams`** (Lookup): `channels=[3,7]`, `min_duration_s`.
- **`DarkPeriods`** (Computed, `-> acquisition.Epoch, -> DarkParams`), with part `Episode(dark_start, dark_end, duration_s, annotation_match)`.
  - Source: `acquisition.LightEvents` over the epoch's chunks. Keyed per epoch because episodes can cross hourly chunks.
  - Start: the first 0 on both channels. End: the next non-zero sample, because logging stops while dark.
  - Cross-check each episode against MessageLog "Dark"/"Light" annotations.
- **`LightPhase`** (Computed, optional): day/dusk-dawn/night intervals from the same stream, as a covariate.
- **`Speed`** (Computed, `-> pose.PositionCM`): store the output of `compute_velocity` on `get_extended_position` (speed in cm/s plus velocity x/y) as xarray. That avoids recomputing it for every PETH.
- **`EarHeading`** (Computed, `-> pose.PoseQC`), implemented 2026-10-24 in `src/dj_pipeline/ovc_behav.py` and `heading.py`:
  - The heading is perpendicular to the left_ear → right_ear axis. Its sign is checked against nose, then head, then spine1, which catches swapped ear labels.
  - Confidence is the ear separation divided by its median.
  - It uses the same QC (`pose_qc.quality_control_pose`) and circular Savitzky–Golay smoothing as foragingABC's `HeadingDirection`, with the same output variables.
  - This is the heading used for OVC. foragingABC's `HeadingDirection` is kept for comparison.
- **`EscapeTrial`** (Computed), with part `Trial` keyed `stim_time`:
  - `key_source`: `Chunk & SoundEvents`, requiring `PositionCM`, `EarHeading` and `Speed` to be populated.
  - Per trigger: `sound_index`, `light_condition` (dark/day/night from `DarkPeriods` and `LightPhase`), `pre_speed`, `peak_speed`, `escape_latency_s`, `reached_shelter`, `heading_to_shelter_deg`, and a −5 to +15 s `trajectory` xarray.
  - Shelter and object fields come from `ArenaLayout` (below). Until that table is ingested, `reached_shelter`, `heading_to_shelter_deg` and `object_condition` stay NULL.
- **`ArenaLayout`** (Manual, **placeholder with `# TODO`**). The shelter and the object are moved during a session and are registered separately, so this table records time-varying layout changes:
  - Master key: `-> acquisition.Experiment, layout_start: datetime(6)`. Fields: `layout_end=null`, `source` (manual / MessageLog / registration file).
  - Part `Shelter`: `shelter_id, x_cm, y_cm, radius_cm`.
  - Part `Object`: `object_id, object_type, x_cm, y_cm`.
  - Add `# TODO: ingest from the shelter/object registration table once its format is final`. Leave a stub `ingest_arena_layout(experiment_name, path)` that raises `NotImplementedError`.
  - `EscapeTrial` looks up the layout active at `stim_time` (`layout_start <= stim_time < layout_end`). Re-populating it after `ArenaLayout` is filled requires deleting the affected `EscapeTrial` rows first. Note this in step07.

### Step 4: Query helpers and runbook
- `AEON_OVC_beh/dj_integration/queries.py`, written in the style of `docs/ephys_runbooks/step06_analysis_examples.py` (deferred imports, `restriction={"experiment_name": ...}`). Every function returns a DataFrame on the HARP clock:
  - `fetch_position(exp, start, end)`, `fetch_heading`, `fetch_speed`
  - `fetch_sound_events(exp)`, `fetch_dark_periods(exp)`, `fetch_escape_trials(exp, light_condition=None)`
- Replace the stub `fetch_behavioral_events` in `step06` with these helpers. Add an example that runs `plot_peth` on units aligned to sound triggers, split by dark vs light.
- Add `step07_behaviour.py` to the runbooks, covering populate order and SLURM usage.

## Critical files
- New:
  - `submodules/foragingABC_analysis` (submodule)
  - `AEON_OVC_beh/dj_integration/{readers.py, ovc_behav.py, queries.py, compat.py, populate_behaviour.py}`
  - `submodules/aeon_mecha/docs/ephys_runbooks/{step01b_behaviour_ingest.py, step07_behaviour.py}`
  - `scripts/hpc/submit_pose.sh`
- Modify: `step06_analysis_examples.py`, `pyproject.toml`, `environment.yml`, `.gitmodules`.
- Reuse:
  - from `setup_pipeline.py`: the streams_maker, DevicesSchema and chunk/EpochConfig steps
  - from foragingABC: `pose.py`, `utils/sleap/*`, `utils/behaviour.py` and `behav.AllMovementBouts`
  - from aeon_mecha: `acquisition.LightEvents` / `MessageLog`, `load_metadata.get_stream_reader_for_epoch`, `fetch_stream`

## Verification
1. Check `dj.list_schemas()` for `aeon_foragingOVC_pose` and `..._ovc_behav`. `dj.Diagram` should show `streams.CameraVideo → pose.* → ovc_behav.EscapeTrial`.
2. On `aeon-ovc-beh-2026-09-28`:
   - `SoundEvents` should hold **28** triggers in total (counts by index: 12→18, 7→14, 6→20, 2→12, 1→16).
   - `DarkPeriods` should have **13** episodes, e.g. 09-29 09:06 (~4 min) and 09-30 16:23–17:01 (~37 min), all matching the MessageLog "Dark" notes.
3. Legacy inference on one hour: `populate_behaviour.py --stage pose --max-calls 1` on a GPU node. The log should show the `sleap_legacy` `sleap-track` binary, and the `.slp` should land in the processed folder. Keypoints should match your earlier legacy analysis `.h5` for that video, if one exists.
4. Pose on one hour:
   - `SleapRawTracking` time length equals the CameraTop frame count. This is already enforced by `harp_time.load_file_frame_timestamps`.
   - `PoseQC` NaN fraction is low.
   - `PositionCM` stays inside the 2 m circle.
   - Speed p99 is under ~150 cm/s.
   - Heading follows the trajectory direction when the animal runs.
5. For one escape listed in MessageLog ("Interesting escape"), `EscapeTrial` shows a short latency and a high peak speed. While `ArenaLayout` is empty, the shelter and object fields are NULL rather than raising errors.
6. On an ephys session, `fetch_escape_trials` + `plot_peth` run end to end, and stimulus and spike times both fall inside the same `EphysBlock` (same HARP clock).
7. Re-running populate inserts nothing new.

## Decisions (confirmed 2026-10-21)
- **Shelter and object positions** come from a separate registration table, to be ingested later. For now, `ArenaLayout` is a placeholder marked TODO and its dependent trial fields are nullable.
- **Calibration:** CameraTop and the arena match foragingABC's AEON3 setup, so the existing calibration is reused.
- **Naming:** ephys and behaviour share the per-session experiment names from step01 (`ProjectAeonOVC-<date>_NNN`).
