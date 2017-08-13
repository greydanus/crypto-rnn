#!/bin/bash -l
#PBS -N enigma
#PBS -q default
#PBS -l nodes=1:gpus=2
#PBS -l walltime=25:30:00
#PBS -l feature="gpu|cent7"
cd /ihome/sgreydan/crypto-rnn/
module load python/2.7-GPU
module load cuda/8.0-cuDNNv6
which python
python main.py --cipher "enigma" --total_steps 1000000  --rnn_size 2048 --tsteps 17 --acc_every 500 --lr 5e-4 --batch_size 50 --key_len 3
