export HF_HOME=/data/nikitha/huggingface/
export CUDA_VISIBLE_DEVICES=5
nohup python3 eval.py --model_id 0 > nohup_llava_7.out 2>&1 &

export CUDA_VISIBLE_DEVICES=6
nohup python3 eval.py --model_id 1 > nohup_llava_ft.out 2>&1 &

export CUDA_VISIBLE_DEVICES=7
nohup python3 eval.py --model_id 2 > nohup_llava_13.out 2>&1 &

export CUDA_VISIBLE_DEVICES=8
nohup python3 eval.py --model_id 3 > nohup_qwen2.out 2>&1 &

export CUDA_VISIBLE_DEVICES=9
nohup python3 eval.py --model_id 4 > nohup_phi3.out 2>&1 &

