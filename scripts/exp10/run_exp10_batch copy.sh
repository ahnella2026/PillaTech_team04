#!/bin/bash
# Exp 10: 3-Seed Training Sequential Batch Script

echo "=== STARTING EXP 10-1 (Seed 42) ==="
python train_yolov8.py --config configs/exp10_seed42.yaml

echo "=== STARTING EXP 10-2 (Seed 123) ==="
python train_yolov8.py --config configs/exp10_seed123.yaml

echo "=== STARTING EXP 10-3 (Seed 777) ==="
python train_yolov8.py --config configs/exp10_seed777.yaml

echo "=== ALL EXP 10 JOBS FINISHED! ==="
