#!/bin/bash
#SBATCH -p short -c 1 --mem=4G -t 03:00:00 -x node08 --array=0-15 -J omol_sel_nfs -o ${HOME}/runs/training/omol_universal/mlmm_ft/logs/select_nfs.%A_%a.out
cd /tmp; export PYTHONPATH=${HOME}/runs/training/omol_universal/pydeps:${HOME}/runs/training/omol_universal/mlmm_ft/scripts
${HOME}/karml-latest/.venv-cu12/bin/python -P ${HOME}/runs/training/omol_universal/mlmm_ft/scripts/select_nfs.py $SLURM_ARRAY_TASK_ID 16
