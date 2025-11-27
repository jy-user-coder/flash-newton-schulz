# On a toy model setup
# python toy_gpt.py --num_steps=300 \
#   --steps_eval_list=0,25,50,100,150,200,300 \
#   --fixed_eval_bs=32 --variant polar_express_aol
# python toy_gpt.py --num_steps=300 \
#   --steps_eval_list=0,25,50,100,150,200,300 \
#   --fixed_eval_bs=32 --variant polar_express_standard
# python toy_gpt.py --num_steps=300 \
#   --steps_eval_list=0,25,50,100,150,200,300 \
#   --fixed_eval_bs=32 --variant aol
# python toy_gpt.py --num_steps=300 \
#   --steps_eval_list=0,25,50,100,150,200,300 \
#   --fixed_eval_bs=32 --variant standard

# On the modded-nanogpt setup
# torchrun --nproc_per_node=4 modded_nanogpt.py --ns_impl dion --save_every 200
torchrun --nproc_per_node=4 modded_nanogpt.py --ns_impl aol --save_every 200
# torchrun --nproc_per_node=4 modded_nanogpt.py --ns_impl std_pe --save_every 200
# torchrun --nproc_per_node=4 modded_nanogpt.py --ns_impl muon --save_every 200

# To get the plot
python plot_gpt.py
