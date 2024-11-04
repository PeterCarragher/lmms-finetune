import json
import numpy as np
import pandas as pd
from transformers import AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoModelForCausalLM 
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from tqdm import tqdm
from eval_utils import *
import random

random.seed(42)

samples_per_image = 5
save = True
eval_data = json.load(open("/data/nikitha/VQA_data/results/WebQA_train_val_obj_v2_generated_labels_shape_color.json", "r"))
perturbation_path = "/data/nikitha/VQA_data/results/old/bad_idx/webqa/"
keys = [k for k in list(eval_data.keys()) if eval_data[k]['split'] == 'val' and eval_data[k]['Qcate'].lower() in ['shape', 'color']]    
keys = keys[:5]
qa_check_df = pd.read_csv('../data/qa_check_perturbation_v4.csv')
qa_check_df = qa_check_df.set_index('file')

model_paths = [
    ("/home/nikithar/Code/VQA/lmms-finetune/checkpoints/llava-1.5-7b_v2_lora-True_qlora-False/", "llava-hf/llava-1.5-7b-hf"), # RET trained
    "llava-hf/llava-1.5-7b-hf",
    "llava-hf/llava-1.5-13b-hf",
    "microsoft/Phi-3-vision-128k-instruct",
    "Qwen/Qwen2-VL-7B-Instruct",

    # single image only:
    # "llava-hf/llava-v1.6-vicuna-7b-hf",
    # "llava-hf/llava-v1.6-vicuna-13b-hf",
    
    # patch size issue:
    # "llava-hf/llava-interleave-qwen-7b-hf",
]

results = {}
exp_name = __file__.split('/')[-1].split('.')[0]

if not os.path.exists("results"):
    os.makedirs("results")
if not os.path.exists(f"results/{exp_name}_answers.json"):
    with open(f"results/{exp_name}_answers.csv", "w") as f:
        f.write("model,question_id,image_id,gen,answer\n")

for model_path in model_paths:
    if isinstance(model_path, tuple):
        model_path, original_model_id = model_path
    else:
        original_model_id = model_path
    print(f"Running evaluation for model: {model_path}")
    
    conversational_prompt = not 'Phi' in model_path
    model, processor = get_model_processor(model_path, original_model_id)

    results_baseline_original = {}
    results_baseline_perturbed= {}
    results_baseline_retrieval_token = {}
    results_perturbed_original = {}
    results_perturbed_perturbed = {}
    results_perturbed_retrieval_token = {}

    for k in tqdm(keys):
        example = eval_data[k]
        if len(example['img_posFacts']) != 2:
            continue
        
        original_image_files = [str(img_data['image_id']) for img_data in example['img_posFacts']]
        try:
            baseline_answer = eval_on_webqa_sample(original_image_files, example, processor, model, conversational_prompt)
        except Exception as e:
            print(f"Error: {e}")
            continue
        
        results_baseline_original[k] = ans_contains_correct_label(baseline_answer, example['A'], example['Qcate'].lower())        
        results_baseline_retrieval_token[k] = retrieval_predicted(baseline_answer)
        
        with open(f"results/{exp_name}_answers.csv", "a") as f:
            f.write(f"{model_path},{k},baseline,baseline,\"{baseline_answer}\"\n")

        results_baseline_perturbed[k] = {}                    
        results_perturbed_original[k] = {}
        results_perturbed_perturbed[k] = {}
        results_perturbed_retrieval_token[k] = {}
        samples_checked = 0
        
        for idx, label in example['A_perturbed'].items():
            idx = int(idx)
            if samples_checked >= samples_per_image:
                break
            imgs = example['img_posFacts']
            if len(imgs) == 2 and idx % 2 == 1:
                continue
            
            results_baseline_perturbed[k][idx] = ans_contains_correct_label(baseline_answer, [label], example['Qcate'].lower())

            try:
                results_perturbed_original[k][idx] = {}
                results_perturbed_perturbed[k][idx] = {}
                results_perturbed_retrieval_token[k][idx] = {}
                
                # TODO: rollback idx change in Segsub repo
                passed_at_least_one_qa_check = False
                for img_idx, img in enumerate(imgs):
                    generated_file = f"{str(img['image_id'])}_{k}_{idx + img_idx}.jpeg"
                    passed_qa_check = file_passes_qa_check(generated_file, qa_check_df)
                    if not passed_qa_check:
                        continue
                    passed_at_least_one_qa_check = True
                    other_file = original_image_files[1 - img_idx]
                    generated_path = os.path.join(perturbation_path, generated_file)
                    generated_image_files = [generated_path, other_file]
                    if img_idx == 1:
                        generated_image_files = generated_image_files[::-1]
                    
                    perturbed_answer = eval_on_webqa_sample(generated_image_files, example, processor, model, conversational_prompt)
                    results_perturbed_original[k][idx][img_idx] = ans_contains_correct_label(perturbed_answer, example['A'], example['Qcate'].lower())
                    results_perturbed_perturbed[k][idx][img_idx] = ans_contains_correct_label(perturbed_answer, [label], example['Qcate'].lower())
                    results_perturbed_retrieval_token[k][idx][img_idx] = retrieval_predicted(perturbed_answer)
                    
                    with open(f"results/{exp_name}_answers.csv", "a") as f:
                        f.write(f"{model_path},{k},{str(original_image_files[img_idx])},{str(idx)},\"{perturbed_answer}\"\n")   

                if passed_at_least_one_qa_check:
                    samples_checked += 1
            except Exception as e:
                print(f"Error: {e}")
                continue
                    
    def recursive_flatten(d):
        flattened_values = []
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, dict):
                    flattened_values.extend(recursive_flatten(v))
                else:
                    flattened_values.append(v)
        return flattened_values
    
    def agg_mean(res):
        return np.mean(recursive_flatten(res))

    results[model_path] = {
        "baseline_original": np.mean(list(results_baseline_original.values())),
        "baseline_ret": np.mean(list(results_baseline_retrieval_token.values())),
        "baseline_perturbed": agg_mean(results_baseline_perturbed),
        "perturbed_original": agg_mean(results_perturbed_original),
        "perturbed_perturbed": agg_mean(results_perturbed_perturbed),
        "perturbed_ret": agg_mean(results_perturbed_retrieval_token),
    }

    with open(f"results/model_outputs/{exp_name}_{model_path.split('/')[-1]}_outputs.json", "w") as f:
        json.dump(results, f)

results_df = pd.DataFrame(results).T
exp_name = __file__.split('/')[-1].split('.')[0]
results_df.to_csv(f"results/{exp_name}_v2.csv")