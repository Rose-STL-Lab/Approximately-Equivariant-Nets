#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python3 run_model.py --relaxed_symmetry=Translation --hidden_dim=128 --num_layers=5 --out_length=6 --num_banks=2 --alpha=0 --batch_size=8 --learning_rate=0.001 --decay_rate=0.95 &
CUDA_VISIBLE_DEVICES=0 python3 run_model.py --relaxed_symmetry=Scale --hidden_dim=64 --num_layers=5 --out_length=6 --alpha=1e-6 --batch_size=8 --learning_rate=0.0001 --decay_rate=0.95 &
CUDA_VISIBLE_DEVICES=0 python3 run_model.py --relaxed_symmetry=Rotation --hidden_dim=92 --num_layers=5 --out_length=6 --alpha=1e-5 --batch_size=16 --learning_rate=0.001 --decay_rate=0.95 &
wait