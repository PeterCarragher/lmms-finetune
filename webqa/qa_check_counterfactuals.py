import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# export LD_LIBRARY_PATH=/home/pcarragh/miniconda3/envs/lmms-finetune/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torchvision

import sys
sys.path.append("..")
from eval.eval_utils import *

eval_data = json.load(open("/home/pcarragh/dev/webqa/MultiModalQA/data/WebQA_train_val_obj_v2.json", "r"))
eval_data = {k: v for k, v in eval_data.items() if v['Qcate'].lower() == 'yesno'}
perturbation_path = "/home/pcarragh/dev/webqa/image_gen_val/val_images_perturbed_gpt_obj_lama"
use_split = False
save = False
keys = list(eval_data.keys())
version = "1"

# model_path = "Qwen/Qwen2-VL-72B-Instruct-AWQ" #
# model_path = "Qwen/Qwen2-VL-7B-Instruct" # 
model_path = "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4"
model, processor = get_model_processor(model_path)
conversational_prompt = not 'Phi' in model_path

results = {}
output_file = "qa_check_counterfactuals"
with open(f"data/{output_file}_v{version}.csv", "w") as f:
    f.write("model,question_id,image_id,qa_check\n")

qa_check_answers = {}
for k in tqdm(keys): 
    example = eval_data[k]
    qa_check_answers[k] = {}
    for img in example['img_posFacts']:
        original_image_file = str(img['image_id'])
        counterfactual_image_file = f"{perturbation_path}/{original_image_file}_{k}.jpeg"
        
        question = f"Q: was the {example['Q_obj']} from the original image removed in the perturbed image?"
        messages = get_qa_check_prompt(question, conversational_prompt)
        try:
            images = get_images([original_image_file, counterfactual_image_file])
        except Exception as e:
            # print(f"Error: {e}")
            continue
        qa_check = run_inference(messages, images, processor, model, conversational_prompt)
        qa_check = qa_check.split('\n')[-1]
        qa_check_answers[k][original_image_file] = qa_check
        with open(f"data/{output_file}_v{version}.csv", "a") as f:
            f.write(f"{model_path},{k},{original_image_file},\"{qa_check}\"\n")

