#!/bin/bash
set -e
bash ${HOME}/karml/run_step_b1_eval_no_crambin.sh
bash ${HOME}/karml/run_step_b2_eval_crambin_sample.sh
