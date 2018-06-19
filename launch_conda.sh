#!/bin/bash
#SBATCH -n 4
#SBATCH --job-name=autoencoder
#SBATCH --mem=10GB
#SBATCH -t 02:30:00
#SBATCH --workdir=/om/user/nprasad/aesap/subs/
#SBATCH --qos=cbmm

python3 /om/user/nprasad/aesap/main.py /om/user/nprasad/aesap/