#!/bin/bash
#YBATCH -r a100_2
#SBATCH --nodes 1
#SBATCH -J synthetic_data
#SBATCH --time=168:00:00
#SBATCH --output outputs/%j.out
#SBATCH --error errors/%j.err

# module load
. /etc/profile.d/modules.sh
module load cuda/11.8
module load cudnn/cuda-11.x/8.9.0

source venv/bin/activate

python exp/gen_ins.py \
  --model_path "google/gemma-2-27b-it" \
  --temperature 1.0 \
  --top_p 1.0 \
  --n 200 \
  --total_prompts 50000 \
  --max_tokens 2048 \
  --max_model_len 4096 \
  --engine vllm \
  --device "0,1" \
  --dtype bfloat16 \
  --tensor_parallel_size 2 \
  --gpu_memory_utilization 0.95 \
  --control_tasks "code" 

echo "Done"