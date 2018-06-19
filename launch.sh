#!/bin/bash
#SBATCH -n 4
#SBATCH --job-name=autoencoder
#SBATCH --mem=10GB
#SBATCH -t 04:30:00
#SBATCH --workdir=/om/user/nprasad/aesap/subs/

singularity exec -B /om:/om --nv /om/user/nprasad/singularity/xboix-localtensorflow.img \
python /om/user/nprasad/aesap/main.py /om/user/nprasad/aesap/