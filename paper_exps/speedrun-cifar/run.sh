## Speedrunning 
CUDA_VISIBLE_DEVICES=0 python airbench94_record.py

## On the convergence / speed bias:
CUDA_VISIBLE_DEVICES=0 python airbench94_polar.py --ns_variant aol_standard --iters_ortho 15 --n_runs 80 --epochs 8
CUDA_VISIBLE_DEVICES=0 python airbench94_polar.py --ns_variant aol_standard --iters_ortho 25 --n_runs 80 --epochs 8
CUDA_VISIBLE_DEVICES=0 python airbench94_polar.py --ns_variant aol_standard --iters_ortho 40 --n_runs 80 --epochs 8
CUDA_VISIBLE_DEVICES=0 python airbench94_polar.py --ns_variant muon_torch --iters_ortho 15 --n_runs 80 --epochs 8
CUDA_VISIBLE_DEVICES=0 python airbench94_polar.py --ns_variant muon_torch --iters_ortho 25 --n_runs 80 --epochs 8
CUDA_VISIBLE_DEVICES=0 python airbench94_polar.py --ns_variant muon_torch --iters_ortho 40 --n_runs 80 --epochs 8
