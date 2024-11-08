import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# export LD_LIBRARY_PATH=/home/pcarragh/miniconda3/envs/lmms-finetune/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
import json
from tqdm import tqdm
import sys
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../webqa/")
from webqa.eval.eval_utils import *
import gc
import argparse
import pandas as pd
import io
import json
import numpy as np
import pandas as pd
from transformers import AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoModelForCausalLM 
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from tqdm import tqdm
import random

save = True
version=3
blank_image_file ='/home/nikithar/Code/VQA/lmms-finetune/webqa/eval/Blank.jpg'
vqa_qid_obj_dir = "/data/nikitha/VQA_data/VQAv2/vqav2_val_obj.txt"
vqa_data = pd.read_feather("/data/nikitha/VQA_data/VQAv2/vqav2_val.arrow")

perturbation_path = "/data/nikitha/VQA_data/VQAv2/results/vqa_removal_val"
qa_check_df = pd.read_csv('../data/qa_check_counterfactual_val.csv')
qa_check_df = qa_check_df.set_index('file')

model_paths = [
    ("/home/nikithar/Code/VQA/lmms-finetune/checkpoints/data_v2/llava-1.5-7b_v2_lora-True_qlora-False/", "llava-hf/llava-1.5-7b-hf"), # RET trained
    # ("/home/nikithar/Code/VQA/lmms-finetune/checkpoints/llava-1.5-7b_v2_lora-True_qlora-False", "llava-hf/llava-1.5-7b-hf"), # RET trained
    # "llava-hf/llava-1.5-7b-hf",
    # "llava-hf/llava-1.5-13b-hf",
    # "microsoft/Phi-3-vision-128k-instruct",
    # "Qwen/Qwen2-VL-7B-Instruct",
]

eval_data = {}
qid_img_q = {}
for _,row in vqa_data.iterrows():
    img = row['image']
    for idx,qid in enumerate(row['question_id']):
        qid_img_q[str(qid)] = {"img": img, "q": row['questions'][idx], "img_id": row['image_id'], "answers": row['answers'][idx]}

with open(vqa_qid_obj_dir, 'r') as f:
    for row in f:
        content = row.rstrip().split('\t')
        assert len(content) == 2
        qid = content[0]
        if qid not in qid_img_q:
            continue
        llm_res = json.loads(content[1])
        eval_data[qid] = {
            'object': llm_res['object'],
            'q': qid_img_q[qid]['q'],
            'img': qid_img_q[qid]['img'],
            'answers': qid_img_q[qid]['answers'],
            # 'new_label': llm_res['new_answer'],
            'qid': qid,
            'image_id': qid_img_q[qid]['img_id']
        }


keys = list(eval_data.keys())
exp_name = __file__.split('/')[-1].split('.')[0] + f"_v{version}"

if not os.path.exists("results"):
    os.makedirs("results")
if not os.path.exists(f"results/{exp_name}_answers.json"):
    with open(f"results/{exp_name}_answers.csv", "w") as f:
        f.write("model,question_id,answer\n")

def ans_contains_correct_vqa_label(answer, labels):
    for label in labels:
        if label.lower() in answer.lower():
            return True
    return False

def get_raw_image(img):
    return Image.open(io.BytesIO(img)).convert("RGB")

results = {}
# prompt_addition = "Answer the following question based only on the provided images. If the image does not containt the answer, respond with 'n/a'.\n"

for model_path in model_paths:
    if isinstance(model_path, tuple):
        model_path, original_model_id = model_path
    else:
        original_model_id = model_path
    print(f"Running evaluation for model: {model_path}")
    
    conversational_prompt = not 'Phi' in model_path
    model, processor = get_model_processor(model_path, original_model_id)

    results_baseline_correct = {}
    results_blank_correct = {}
    results_perturbed_correct = {}
    results_baseline_retrieval_token = {}
    results_blank_retrieval_token = {}
    results_perturbed_retrieval_token = {}

    for k in tqdm(keys):
        example = eval_data[k]

        try:
            original_image = get_raw_image(example['img'])
            generated_file = f"{perturbation_path}/{str(example['image_id'])}_{k}.jpeg"
            if not os.path.exists(generated_file) or not file_passes_qa_check(generated_file, qa_check_df):
                continue
            
            baseline_answer = eval_on_vqa_sample(original_image, example['q'], processor, model, conversational_prompt)            
            blank_answer = eval_on_vqa_sample(blank_image_file, example['q'], processor, model, conversational_prompt)
            perturbed_answer = eval_on_vqa_sample(generated_file, example['q'], processor, model, conversational_prompt)
        except Exception as e:
            print(f"Error: {e}")
            continue
                
        results_baseline_correct[k] = ans_contains_correct_vqa_label(baseline_answer, example['answers'])
        results_blank_correct[k] = ans_contains_correct_vqa_label(blank_answer, example['answers'])
        results_perturbed_correct[k] = ans_contains_correct_vqa_label(perturbed_answer, example['answers'])
        results_baseline_retrieval_token[k] = retrieval_predicted(baseline_answer)
        results_blank_retrieval_token[k] = retrieval_predicted(blank_answer)
        results_perturbed_retrieval_token[k] = retrieval_predicted(perturbed_answer)
        
        with open(f"results/{exp_name}_answers.csv", "a") as f:
            f.write(f"{model_path},{k},\"{perturbed_answer}\"\n")
 
    results[model_path] = {
        "baseline_correct": np.mean(list(results_baseline_correct.values())),
        "baseline_ret": np.mean(list(results_baseline_retrieval_token.values())),
        "blank_correct": np.mean(list(results_blank_correct.values())),
        "blank_ret": np.mean(list(results_blank_retrieval_token.values())),
        "perturbed_correct": np.mean(list(results_perturbed_correct.values())),
        "perturbed_ret": np.mean(list(results_perturbed_retrieval_token.values()))
    }
    print(results[model_path])
    
    if not os.path.exists("results/model_outputs"):
        os.makedirs("results/model_outputs")
    with open(f"results/model_outputs/{exp_name}_{model_path.split('/')[-1]}_outputs.json", "w") as f:
        json.dump(results, f)

results_df = pd.DataFrame(results).T
exp_name = __file__.split('/')[-1].split('.')[0]
results_df.to_csv(f"results/{exp_name}.csv")