CUDA_VISIBLE_DEVICES=0 python airbench94.py --ns_variant aol_standard --iters_ortho 4 --n_runs 20 --epochs 7.5
CUDA_VISIBLE_DEVICES=0 python airbench94.py --ns_variant muon_triton --iters_ortho 3 --n_runs 20 --epochs 8
CUDA_VISIBLE_DEVICES=0 python airbench94.py --ns_variant aol_conv --iters_ortho 3 --n_runs 5 --epochs 8
CUDA_VISIBLE_DEVICES=0 python airbench94.py --ns_variant muon_torch --iters_ortho 3 --n_runs 5 --epochs 8
CUDA_VISIBLE_DEVICES=0 python airbench94.py --ns_variant dion --iters_ortho 3 --n_runs 5 --epochs 8
