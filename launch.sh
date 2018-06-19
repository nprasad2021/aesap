#!/bin/bash
#SBATCH -n 4
#SBATCH --job-name=autoencoder
#SBATCH --mem=10GB
#SBATCH -t 02:30:00
#SBATCH --gres=gpu:1
#SBATCH --workdir=/om/user/nprasad/aesap/subs/
#SBATCH --qos=cbmm

singularity exec -B /om:/om --nv /om/user/nprasad/singularity/tensorflow-1.8.0-gpu-py3.img \
python /om/user/nprasad/aesap/main.py /om/user/nprasad/aesap/