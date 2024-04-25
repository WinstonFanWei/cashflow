#!/bin/sh
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -n 2
#BSUB -q gpu
#BSUB -o out_err/%J.out
#BSUB -e out_err/%J.error_out
#BSUB -J gputest
#BSUB -R "rusage[ngpus_physical=2]"
python Main.py
