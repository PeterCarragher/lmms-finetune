import json
import numpy as np
import pandas as pd

from tqdm import tqdm
import sys
sys.path.append("..")

from eval.eval_utils import *

# eval_data = json.load(open("/home/pcarragh/dev/webqa/LLaVA/WebQA_train_val_color_gpt_matched.json", "r"))
eval_data = json.load(open("/home/pcarragh/dev/webqa/MultiModalQA/data/WebQA_train_val_obj_v2.json.pert.lama.v3", "r"))
eval_data = {k: v for k, v in eval_data.items() if v['Qcate'].lower() in ['shape', 'color']}
perturbation_path = "/home/pcarragh/dev/webqa/segment/Inpaint-Anything/results/webqa"
use_split = True
save = False

keys = list(eval_data.keys())
version = "1"

# model_path = "Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4"
# model_path = "Qwen/Qwen2-VL-7B-Instruct" 
model_path = "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4" 

model, processor = get_model_processor(model_path)
conversational_prompt = not 'Phi' in model_path

results = {}
output_file = "qa_check_perturbations"
with open(f"data/{output_file}_v{version}.csv", "w") as f:
    f.write("model,question_id,image_id,gen_id,qa_check\n")

qa_check_answers = {}
for k in tqdm(keys):
    example = eval_data[k]
    qa_check_answers[k] = {}
    for img in example['img_posFacts']:
        original_image_file = str(img['image_id'])
        qcate = example['Qcate'].lower()
        orig_answer = example['A'][0]
        orig_answer = normalize_text(orig_answer)
        
        # TODO: does getting first term mess up QA check for answers with multiple labels?
        orig_answer = find_first_search_term(orig_answer, domain_dict[qcate], qcate, orig_answer)
        qa_check_answers[k][original_image_file] = {}
        
        # TODO: on server, we have perturbations for both images, need another index
        if not 'A_perturbed' in example:
            continue
        for idx, label in example['A_perturbed'].items():
            counterfactual_image_file = f"{perturbation_path}/{example['split']}/{original_image_file}_{k}_{idx}.jpeg"
            
            question = f"Q: is the {qcate} of the {example['Q_obj']} the same in both images?"
            # question = f"Q: has the {qcate} of the {example['Q_obj']} changed from {orig_answer} in the original image to {label} in the perturbed image?"
            messages = get_qa_check_prompt(question, conversational_prompt)
            try:
                images = get_images([original_image_file, counterfactual_image_file])
            except Exception as e:
                print(f"Error: {e}")
                continue
            qa_check = run_inference(messages, images, processor, model, conversational_prompt)
            qa_check = qa_check.split('\n')[-1]
            qa_check_answers[k][original_image_file][idx] = qa_check
            with open(f"data/{output_file}_v{version}.csv", "a") as f:
                f.write(f"{model_path},{k},{original_image_file},{idx},\"{qa_check}\"\n")
                    