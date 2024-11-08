import json
import numpy as np
import pandas as pd
from transformers import AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoModelForCausalLM 
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from tqdm import tqdm
from eval_utils import *
import random

random.seed(42)

version = 3
samples_per_image = 5
save = True
eval_data = json.load(open("/data/nikitha/VQA_data/results/WebQA_train_val_obj_v2_generated_labels_shape_color.json", "r"))
perturbation_path = "/data/nikitha/VQA_data/results/old/bad_idx/webqa/"
keys = [k for k in list(eval_data.keys()) if eval_data[k]['split'] == 'val' and eval_data[k]['Qcate'].lower() in ['shape', 'color']]    
qa_check_df = pd.read_csv('../data/qa_check_perturbation_v4.csv')
qa_check_df = qa_check_df.set_index('file')


model_paths = [
    ("/home/nikithar/Code/VQA/lmms-finetune/checkpoints/data_v2/llava-1.5-7b_v2_lora-True_qlora-False/", "llava-hf/llava-1.5-7b-hf"), # RET trained
    # ("/home/nikithar/Code/VQA/lmms-finetune/checkpoints/llava-1.5-7b_v2_lora-True_qlora-False/", "llava-hf/llava-1.5-7b-hf"), # RET trained
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
exp_name = __file__.split('/')[-1].split('.')[0] + f"_v{version}"

if not os.path.exists("results"):
    os.makedirs("results")
if not os.path.exists(f"results/{exp_name}_answers.json"):
    with open(f"results/{exp_name}_answers.csv", "w") as f:
        f.write("model,question_id,gen,answer\n")

for model_path in model_paths:
    if isinstance(model_path, tuple):
        model_path, original_model_id = model_path
    else:
        original_model_id = model_path
    print(f"Running evaluation for model: {model_path}")
    
    conversational_prompt = not 'Phi' in model_path
    model, processor = get_model_processor(model_path, original_model_id)

    results_baseline_original_label = {}
    results_baseline_perturbed_label = {}
    results_perturbed_original_label = {}
    results_perturbed_perturbed_label = {}

    for k in tqdm(keys):
        example = eval_data[k]
        original_image_files = [str(img_data['image_id']) for img_data in example['img_posFacts']]
        try:
            baseline_answer = eval_on_webqa_sample(original_image_files, example, processor, model, conversational_prompt)
        except Exception as e:
            print(f"Error: {e}")
            continue
        
        results_baseline_original_label[k] = webqa_accuracy(baseline_answer, example['A'], example['Qcate'].lower())        

        with open(f"results/{exp_name}_answers.csv", "a") as f:
            f.write(f"{model_path},{k},baseline,\"{baseline_answer}\"\n")
                    
        results_perturbed_perturbed_label[k] = {}
        results_perturbed_original_label[k] = {}
        results_baseline_perturbed_label[k] = {}
        samples_checked = 0
        for idx, label in example['A_perturbed'].items():
            idx = int(idx)
            if samples_checked >= samples_per_image:
                break
            imgs = example['img_posFacts']
            if len(imgs) == 2 and idx % 2 == 1:
                continue

            try:
                generated_image_files = []
                for hack_idx, img in enumerate(imgs):
                    generated_file = f"{str(img['image_id'])}_{k}_{idx + hack_idx}.jpeg"
                    generated_path = os.path.join(perturbation_path, generated_file)
                    generated_image_files.append(generated_path)

                if any([not file_passes_qa_check(file, qa_check_df) for file in generated_image_files]):
                    continue
                perturbed_answer = eval_on_webqa_sample(generated_image_files, example, processor, model, conversational_prompt)
                samples_checked += 1
            except Exception as e:
                print(f"Error: {e}")
                continue
            
            results_perturbed_original_label[k][idx] = webqa_accuracy(perturbed_answer, example['A'], example['Qcate'].lower())
            results_perturbed_perturbed_label[k][idx] = webqa_accuracy(perturbed_answer, [label], example['Qcate'].lower())
            results_baseline_perturbed_label[k][idx] = webqa_accuracy(baseline_answer, [label], example['Qcate'].lower())
            
            with open(f"results/{exp_name}_answers.csv", "a") as f:
                f.write(f"{model_path},{k},{str(idx)},\"{perturbed_answer}\"\n")
               
    results[model_path] = {
        "baseline_original_label": accuracy_agg_results(results_baseline_original_label, eval_data),
        "baseline_perturbed_label": accuracy_agg_generated_results(results_baseline_perturbed_label, eval_data),
        "perturbed_original_label": accuracy_agg_generated_results(results_perturbed_original_label, eval_data),
        "perturbed_perturbed_label": accuracy_agg_generated_results(results_perturbed_perturbed_label, eval_data)
    }
    
    with open(f"results/model_outputs/{exp_name}_{model_path.split('/')[-1]}_outputs.json", "w") as f:
        json.dump(results, f)

formatted = {}
for model_path, result in results.items():
    formatted[model_path + '_1_image'] = {k:v for (k, (v, _, _)) in result.items()}
    formatted[model_path + '_2_image'] = {k:v for (k, (_, v, _)) in result.items()}
    formatted[model_path + '_N_image'] = {k:v for (k, (_, _, v)) in result.items()}
    # print(f"Model: {model_path}")
    # print(result)


formatted_df = pd.DataFrame(formatted).T

if not save:
    print(formatted_df)
elif not os.path.exists(f"results/{exp_name}.csv"):
    formatted_df.to_csv(f"results/{exp_name}.csv")
else:
    formatted_df.to_csv(f"results/{exp_name}.csv", mode='a', header=False)