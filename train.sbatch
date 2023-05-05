#!/bin/bash
#SBATCH --job-name="Optimise lightfield"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:8,VRAM:12G
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
python trainval.py -e pointLF_waymo_local -sb ~/storage/user/output -d ~/storage/user/data/ --epoch_size 100 --num_workers 4 --mask True --ngpus 8 --render_only True