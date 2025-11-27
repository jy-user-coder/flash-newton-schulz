# normal + uniform
python benchmark.py --dims 64 128 256 512 1024 2048 --dtype bfloat16 --batch 32 --warmup 2 --rep 4 --dist normal uniform --iters 1 2 3 4 5 --csv results_1/0_normal_unif.csv

# normal + uniform
python benchmark.py --dims 4096 8192 --dtype bfloat16 --batch 8 --warmup 2 --rep 4 --dist normal uniform --iters 1 2 3 4 5 --csv results_1/0_normal_unif_xl.csv

# levy distribution
python benchmark.py --dims 64 128 256 512 1024 --dtype bfloat16 --batch 4 --warmup 2 --rep 4 --dist levy --levy-alpha 1.0 --iters 1 2 3 4 5 --csv results_1/0_levy-1-0.csv

python benchmark.py --dims 64 128 256 512 1024 --dtype bfloat16 --batch 4 --warmup 2 --rep 4 --dist levy --levy-alpha 1.5 --iters 1 2 3 4 5 --csv results_1/0_levy-1-5.csv

python benchmark.py --dims 64 128 256 512 1024 --dtype bfloat16 --batch 4 --warmup 2 --rep 4 --dist levy --levy-alpha 2.0 --iters 1 2 3 4 5 --csv results_1/0_levy-2-0.csv


# normal + uniform
python benchmark.py --dims 64 128 256 512 1024 2048 --dtype float32 --batch 32 --warmup 2 --rep 4 --dist normal uniform --iters 1 2 3 4 5 --csv results_1/0_normal_unif_f32.csv

# normal + uniform
python benchmark.py --dims 4096 8192 --dtype float32 --batch 8 --warmup 2 --rep 4 --dist normal uniform --iters 1 2 3 4 5 --csv results_1/0_normal_unif_xl_f32.csv
