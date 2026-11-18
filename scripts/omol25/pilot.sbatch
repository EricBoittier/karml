#!/bin/bash
#SBATCH -p gpu --gres=gpu:1 -c 4 --mem=24G -t 24:00:00 -J omol_pilot -o ${HOME}/runs/training/omol_universal/logs/pilot.%j.out
# OMol25 universal SpookyPhysNet pilot (200k structures, F64 l1 i3, cutoff 6 A, charge + spin conditioned).
cd /tmp
export OMPI_MCA_orte_tmpdir_base=$(mktemp -d /tmp/ompi_job.XXXXXX)
${HOME}/karml-latest/.venv-cu12/bin/python -I ${HOME}/runs/training/omol_universal/scripts/launch_train.py \
  ${HOME}/runs/training/omol_universal/configs/pilot.yaml --exec
