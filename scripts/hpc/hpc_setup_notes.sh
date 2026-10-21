# HPC setup notes
# ================
# Run these commands on the HPC LOGIN NODE before submitting your first job.

# ---- 1. Clone the repo onto HPC ----
cd ~
git clone https://github.com/yourlab/np2_repo.git   # or scp/rsync from local
cd np2_repo

# ---- 2. Create conda environment ----
# If conda not available, load it first: module load miniconda/4.x
conda env create -f environment.yml
conda activate np2_pipeline

# Install Kilosort4 with GPU (CUDA 11.8 — match your cluster's CUDA module)
pip install kilosort
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Register the kernel for Jupyter (if you want to run notebooks on HPC)
python -m ipykernel install --user --name np2_pipeline --display-name "NP2 pipeline"

# ---- 3. Pull Singularity image for KS2.5 ----
# This only needs to be done once. Store it somewhere persistent.
mkdir -p ~/singularity_images
cd ~/singularity_images

# Pull from Docker Hub (internet access needed — check HPC policy)
singularity pull kilosort2_5.sif docker://spikeinterface/kilosort2_5-compiled-base

# If the login node has no internet, build locally and transfer:
#   LOCAL: docker pull spikeinterface/kilosort2_5-compiled-base
#   LOCAL: docker save spikeinterface/kilosort2_5-compiled-base | gzip > ks25.tar.gz
#   LOCAL: scp ks25.tar.gz myriad.rc.ucl.ac.uk:~/singularity_images/
#   HPC:   singularity build kilosort2_5.sif docker-archive:///~/singularity_images/ks25.tar.gz

# ---- 4. Check your scratch/Ceph quotas ----
# On Myriad:
lquota                          # home + Scratch quota
du -sh /ceph/neuroinformatics/rokas/   # Ceph usage

# For a 24 hr NP2.0 recording you need roughly:
#   2 TB  — raw (already on Ceph)
#   2 TB  — preprocessed float32 binary (scratch)
#   2 TB  — KS temp during sorting (scratch, deleted after)
#   15 GB — SortingAnalyzer waveforms (copied to Ceph at end)
# = ~6 TB scratch peak usage

# Request tmpfs on Myriad with #$ -l tmpfs=6T in the job script.
# If the node tmpfs is smaller, use a persistent scratch path instead:
#   SCRATCH_BASE=/scratch/scratch/$USER/np2_session

# ---- 5. Test the environment interactively before a long job ----
# Request a short interactive GPU job:
#   SLURM:  srun --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G \
#                --gres=gpu:1 --partition=gpu --time=1:00:00 --pty bash
#   SGE:    qrsh -pe smp 4 -l mem=8G,gpu=1,h_rt=1:00:00 -l tmpfs=100G

# Then run the pipeline with --skip-sorting on a 5-minute subset:
python - <<'EOF'
import sys
sys.path.insert(0, 'src')
import spikeinterface.full as si
from np2_pipeline.io import load_onix_recording

rec = load_onix_recording(
    raw_dir='/ceph/neuroinformatics/rokas/data/raw/session_id',
    n_channels=384, dtype='uint16', gain_to_uV=2.34375
)
# Test reading 5 min
rec_sub = rec.frame_slice(0, int(5 * 60 * 30000))
rec_hp = si.highpass_filter(rec_sub, freq_min=300)
# Force materialisation of a small chunk
trace = rec_hp.get_traces(start_frame=0, end_frame=30000)
print(f"Read ok: shape {trace.shape}, dtype {trace.dtype}")
EOF

# ---- 6. Monitor a running job ----
#   SLURM:  squeue -u $USER
#           tail -f logs/np2_JOBID.out
#           tail -f /scratch/scratch/$USER/np2_SESSION/pipeline.log
#
#   SGE:    qstat -u $USER
#           tail -f logs/np2_JOBID.out

# ---- 7. Submit ----
#   SLURM:  sbatch scripts/hpc/submit_job.sh
#   SGE:    qsub scripts/hpc/submit_job.sh
