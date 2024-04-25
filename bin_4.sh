#!/bin/sh
#BSUB -gpu "num=4:mode=exclusive_process"
#BSUB -n 4
#BSUB -q gpu
#BSUB -o out_err/%J.out
#BSUB -e out_err/%J.error_out
#BSUB -J gputest
#BSUB -R "rusage[ngpus_physical=2] span[ptile=2]"
python Main.py
