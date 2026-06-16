import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────
#Paths to SLEAP outputs and Environmet folders

SLEAP_OUTPUTS_FOLDER = '/run/user/1000/gvfs/smb-share:server=ceph-gw02.hpc.swc.ucl.ac.uk,share=branco/Rokas/Ephys/Results/Behaviour/2025-03-26_001/sleap_predictions'
FRAME_INFO_FOLDER = '/run/user/1000/gvfs/smb-share:server=ceph-gw02.hpc.swc.ucl.ac.uk,share=aeon/test2/scratchpad/2026-03-26T130404Z/CameraTop'
ENVIRONMENT_FOLDER = '/run/user/1000/gvfs/smb-share:server=ceph-gw02.hpc.swc.ucl.ac.uk,share=aeon/test2/scratchpad/2026-03-26T130404Z/Environment'

OUTPUT_DIR = '/run/user/1000/gvfs/smb-share:server=ceph-gw02.hpc.swc.ucl.ac.uk,share=branco/Rokas/Ephys/Results/Behaviour/2025-03-26_001/concatenated_behaviour'

# ─── CONCATENATE SLEAP OUTPUTS ───────────────────────────────────────────────────────────
def concatenate_sleap_outputs(sleap_folder, output_dir):
    # List all SLEAP output files
    sleap_files = sorted(Path(sleap_folder).glob("*.h5"))
    if len(sleap_files) == 0:
        print(f"No SLEAP .h5 files found in {sleap_folder}")
        return

    all_tracks = []
    all_point_scores = []
    node_names_ref = None

    for file in sleap_files:
        with h5py.File(file, 'r') as f:
            # expected datasets: tracks (n_tracks, 2, n_nodes, n_frames)
            tracks = f['tracks'][:]
            point_scores = f['point_scores'][:]
            raw_node_names = f['node_names'][:]
            node_names = [n.decode() if isinstance(n, (bytes, np.bytes_)) else str(n) for n in raw_node_names]

            if node_names_ref is None:
                node_names_ref = node_names
            elif node_names != node_names_ref:
                print(f"Warning: node_names differ in {file}; using first file's node_names")

            all_tracks.append(tracks)
            all_point_scores.append(point_scores)

    # Concatenate along the frames axis (last axis)
    concatenated_tracks = np.concatenate(all_tracks, axis=-1)
    concatenated_point_scores = np.concatenate(all_point_scores, axis=-1)

    # Save concatenated datasets to a new HDF5 file
    output_path = Path(output_dir) / "concatenated_predictions.h5"
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('tracks', data=concatenated_tracks)
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('node_names', data=np.array(node_names_ref, dtype=dt))
        f.create_dataset('point_scores', data=concatenated_point_scores)

    print(f"Concatenated SLEAP outputs saved to {output_path}")

# ─── CONCATENATE FRAME ALIGNED DATA ───────────────────────────────────────────────────────────────────
def concatenate_frame_info(frame_info_folder, output_dir):
    # List all environment state files with Environment_EnvironmentState prefix
    frame_info_files = sorted(Path(frame_info_folder).glob("CameraTop*.csv"))

    all_frame_info = []
    for file in frame_info_files:
        frame_info = pd.read_csv(file)
        all_frame_info.append(frame_info)
    
    # Concatenate all frame info along the rows (time)
    concatenated_frame_info = pd.concat(all_frame_info, ignore_index=True)
    
    # Save concatenated frame info to a new CSV file
    output_path = Path(output_dir) / "concatenated_frame_info.csv"
    concatenated_frame_info.to_csv(output_path, index=False)
    
    print(f"Concatenated frame info saved to {output_path}")

# ─── CONCATENATE ENVIRONMENT STATE ───────────────────────────────────────────────────────────────────
def concatenate_environment_state(environment_folder, output_dir):
    # List all environment state files with Environment_EnvironmentState prefix
    env_files = sorted(Path(environment_folder).glob("Environment_EnvironmentState*.csv"))
    
    all_env_data = []
    for file in env_files:
        env_data = pd.read_csv(file)
        all_env_data.append(env_data)
    
    # Concatenate all environment data along the rows (time)
    concatenated_env_data = pd.concat(all_env_data, ignore_index=True)
    
    # Save concatenated environment data to a new CSV file
    output_path = Path(output_dir) / "concatenated_environment_state.csv"
    concatenated_env_data.to_csv(output_path, index=False)
    
    print(f"Concatenated environment state saved to {output_path}")

# ─── CONCATENATE MESSAGE LOG ─────────────────────────────────────────────────────────────
def concatenate_message_log(message_log_folder, output_dir):
    # List all message log files with Environment_MessageLog prefix
    env_files = sorted(Path(message_log_folder).glob("Environment_MessageLog*.csv"))
    
    all_message_data = []
    for file in env_files:
        env_data = pd.read_csv(file)
        all_message_data.append(env_data)
    
    # Concatenate all message data along the rows (time)
    concatenated_message_data = pd.concat(all_message_data, ignore_index=True)
    
    # Save concatenated message data to a new CSV file
    output_path = Path(output_dir) / "concatenated_message_log.csv"
    concatenated_message_data.to_csv(output_path, index=False)
    
    print(f"Concatenated message Log saved to {output_path}")

# ─── MAIN FUNCTION ─────────────────────────────────────────────────────────────
def main():
    # Create output directory if it doesn't exist
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Concatenate SLEAP outputs
    concatenate_sleap_outputs(SLEAP_OUTPUTS_FOLDER, OUTPUT_DIR)

    # Concatenate frame info files
    concatenate_frame_info(FRAME_INFO_FOLDER, OUTPUT_DIR)
    
    # Concatenate environment state files
    concatenate_environment_state(ENVIRONMENT_FOLDER, OUTPUT_DIR)
    
    # Concatenate message log files
    concatenate_message_log(ENVIRONMENT_FOLDER, OUTPUT_DIR)

if __name__ == "__main__":
    main()