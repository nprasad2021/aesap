#!/bin/bash
#SBATCH --qos=cbmm
#SBATCH --time=02:00:00
#SBATCH --job-name=tbjob
#SBATCH --output=tbjob%j.out
#SBATCH --error=tbjob%j.err

singularity exec -B /om:/om /om/user/nprasad/singularity/tensorflow-1.8.0-gpu-py3.img tensorboard \
--port=8022 \
--logdir=/om/user/nprasad/aesap/log/initial/ID0_base/