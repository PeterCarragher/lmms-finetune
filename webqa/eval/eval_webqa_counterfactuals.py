import json
import numpy as np
import pandas as pd
from transformers import AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoModelForCausalLM 
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from tqdm import tqdm
from eval_utils import *
import random

version = 3
save = True
blank_image_file ='/home/nikithar/Code/VQA/lmms-finetune/webqa/eval/Blank.jpg'
eval_data = json.load(open("/data/nikitha/VQA_data/WebQA_train_val_obj_v2.json", "r"))
perturbation_path = "/data/nikitha/VQA_data/results/webqa_yesno/"
keys = [k for k in list(eval_data.keys()) if eval_data[k]['split'] == 'val' and eval_data[k]['Qcate'].lower() in ['yesno']]    
# keys = keys[:10]
qa_check_df = pd.read_csv('../data/qa_check_counterfactuals_v2.csv')
qa_check_df = qa_check_df.set_index('file')

model_paths = [
    ("/home/nikithar/Code/VQA/lmms-finetune/checkpoints/data_v2/llava-1.5-7b_v2_lora-True_qlora-False/", "llava-hf/llava-1.5-7b-hf"), # RET trained
    # ("/home/nikithar/Code/VQA/lmms-finetune/checkpoints/llava-1.5-7b_v2_lora-True_qlora-False", "llava-hf/llava-1.5-7b-hf"), # RET trained
    # "llava-hf/llava-1.5-7b-hf",
    # "llava-hf/llava-1.5-13b-hf",
    # "microsoft/Phi-3-vision-128k-instruct",
    # "Qwen/Qwen2-VL-7B-Instruct",
    # single image only:
    # "llava-hf/llava-v1.6-vicuna-7b-hf",
    # "llava-hf/llava-v1.6-vicuna-13b-hf",
    
    # patch size issue:
    # "llava-hf/llava-interleave-qwen-7b-hf",
]

results = {}
exp_name = __file__.split('/')[-1].split('.')[0]

answer_file =f"results/{exp_name}_answers_v{version}.csv"
if not os.path.exists("results"):
    os.makedirs("results")
if not os.path.exists(f"results/{exp_name}_answers.json"):
    with open(answer_file, "w") as f:
        f.write("model,question_id,gen,answer\n")

results = {}
# prompt_addition = "Answer the following question based only on the provided images. If the image does not containt the answer, respond with 'n/a'.\n"

for model_path in model_paths:
    if isinstance(model_path, tuple):
        model_path, original_model_id = model_path
    else:
        original_model_id = model_path
    print(f"Running evaluation for model: {model_path}")
    
    conversational_prompt = not 'Phi' in model_path
    if model_path.endswith(".jsonl"):
        # read answer file instead of loading model
        model = None
        processor = None
        answers = json.load(open(model_path, "r"))
        val_set = json.load(open("../data/webqa_val_gen_formatted_v2.json", "r"))
        
    else:
        model, processor = get_model_processor(model_path, original_model_id)

    results_baseline_correct = {}
    results_baseline_any = {}
    results_blank_correct = {}
    results_blank_any = {}
    results_perturbed_correct = {}
    results_perturbed_any = {}
    results_baseline_retrieval_token = {}
    results_blank_retrieval_token = {}
    results_perturbed_retrieval_token = {}

    for k in tqdm(keys):
        example = eval_data[k]
        original_image_files = [str(img_data['image_id']) for img_data in example['img_posFacts']]
        blank_image_files = [blank_image_file for _ in example['img_posFacts']]
        generated_image_files = []
        # question = prompt_addition + get_prompt(example)

        try:
            for img in example['img_posFacts']:
                generated_file = f"{perturbation_path}/{str(img['image_id'])}_{k}.jpeg"
                generated_image_files.append(generated_file)
                # if os.path.exists(generated_file):
                #     generated_image_files.append(generated_file)
                # else:
                    # generated_image_files.append(str(img['image_id']))
            
            if any([not file_passes_qa_check(file, qa_check_df) for file in generated_image_files]):
                continue
              
            baseline_answer = eval_on_webqa_sample(original_image_files, example, processor, model, conversational_prompt)
            blank_answer = eval_on_webqa_sample(blank_image_files, example, processor, model, conversational_prompt)
            perturbed_answer = eval_on_webqa_sample(generated_image_files, example, processor, model, conversational_prompt)
        except Exception as e:
            print(f"Error: {e}")
            continue
                
        results_baseline_correct[k] = ans_contains_correct_label(baseline_answer, example['A'], example['Qcate'].lower())
        results_baseline_any[k] = ans_contains_any_label(baseline_answer)
        results_blank_correct[k] = ans_contains_correct_label(blank_answer, example['A'], example['Qcate'].lower())
        results_blank_any[k] = ans_contains_any_label(blank_answer)
        results_perturbed_correct[k] = ans_contains_correct_label(perturbed_answer, example['A'], example['Qcate'].lower())
        results_perturbed_any[k] = ans_contains_any_label(perturbed_answer)
        results_baseline_retrieval_token[k] = retrieval_predicted(baseline_answer)
        results_blank_retrieval_token[k] = retrieval_predicted(blank_answer)
        results_perturbed_retrieval_token[k] = retrieval_predicted(perturbed_answer)
        
        with open(answer_file, "a") as f:
            f.write(f"{model_path},{k},\"{perturbed_answer}\"\n")
 
    results[model_path] = {
        "baseline_correct": np.mean(list(results_baseline_correct.values())),
        "baseline_any": np.mean(list(results_baseline_any.values())),
        "baseline_ret": np.mean(list(results_baseline_retrieval_token.values())),
        "blank_correct": np.mean(list(results_blank_correct.values())),
        "blank_any": np.mean(list(results_blank_any.values())),
        "blank_ret": np.mean(list(results_blank_retrieval_token.values())),
        "perturbed_correct": np.mean(list(results_perturbed_correct.values())),
        "perturbed_any": np.mean(list(results_perturbed_any.values())),
        "perturbed_ret": np.mean(list(results_perturbed_retrieval_token.values()))
    }
    print(results[model_path])
    
    with open(f"results/model_outputs/{exp_name}_{model_path.split('/')[-1]}_outputs.json", "w") as f:
        json.dump(results, f)

results_df = pd.DataFrame(results).T
exp_name = __file__.split('/')[-1].split('.')[0]
results_df.to_csv(f"results/{exp_name}_v{version}.csv")