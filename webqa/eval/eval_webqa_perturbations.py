import json
import numpy as np
import pandas as pd
from transformers import AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoModelForCausalLM 
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from tqdm import tqdm
from eval_utils import *

# single image = perturbation, 2 image = conflicting
eval_data = json.load(open("/home/pcarragh/dev/webqa/LLaVA/WebQA_train_val_color_gpt_matched.json", "r"))
blank_image_file ='/home/pcarragh/dev/webqa/LLaVA/playground/counterfactual_exp/BLANK.jpg'
perturbation_path = "/home/pcarragh/dev/webqa/segment/Inpaint-Anything/results/webqa"
use_split = True
save = False
keys = list(eval_data.keys())
# qids = pd.read_csv('results/counterfactual_qa_check.csv', header=None)[0].tolist()
# eval_data = {k: v for k, v in eval_data.items() if k in qids}

model_paths = [
    "llava-hf/llava-1.5-7b-hf",
    "llava-hf/llava-1.5-13b-hf",
    "Qwen/Qwen2-VL-7B-Instruct",
    "microsoft/Phi-3-vision-128k-instruct",

    # single image only:
    # "llava-hf/llava-v1.6-vicuna-7b-hf",
    # "llava-hf/llava-v1.6-vicuna-13b-hf",
    
    # patch size issue:
    # "llava-hf/llava-interleave-qwen-7b-hf",
]


results = {}
# answers.csv dump file
with open("results/answers.csv", "w") as f:
    f.write("model,question_id,gen,answer\n")

for model_path in model_paths:
    print(f"Running evaluation for model: {model_path}")
    
    original_model_id = model_path 
    
    # for finetuning...
    # model_id = "/home/pcarragh/dev/webqa/lmms-finetune/checkpoints/llava-1.5-7b_lora-True_qlora-False/"
    # model_path = model_id
    
    conversational_prompt = True
    if "llava-v1.6" in model_path or "llava-1.6" in model_path:
        processor = LlavaNextProcessor.from_pretrained(model_path)
        model = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")#, low_cpu_mem_usage=True) 
    elif "llava" in model_path:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path, # model_id, 
            torch_dtype=torch.float16, 
            # low_cpu_mem_usage=True, 
        ).to("cuda")
        processor = AutoProcessor.from_pretrained(original_model_id)
    elif "Qwen" in model_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_path)
    elif "Phi" in model_path:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True, torch_dtype="auto", _attn_implementation='flash_attention_2') 
        # use _attn_implementation='eager' to disable flash attention
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True) 
        conversational_prompt = False
    else:
        raise ValueError(f"Unknown model path: {model_path}")

    llava_results_baseline_original_label = {}
    llava_results_baseline_perturbed_label = {}
    llava_results_blank_original_label = {}
    llava_results_blank_perturbed_label = {}
    llava_results_perturbed_original_label = {}
    llava_results_perturbed_perturbed_label = {}

    for k in tqdm(keys):
        example = eval_data[k]
        if len(example['img_posFacts']) != 2:
            continue
        original_image_files = [str(img_data['image_id']) for img_data in example['img_posFacts']]
        blank_image_files = [blank_image_file for _ in example['img_posFacts']]
        try:
            baseline_answer = eval_on_webqa_sample(original_image_files, example, processor, model, conversational_prompt)
            blank_answer = eval_on_webqa_sample(blank_image_files, example, processor, model, conversational_prompt)
        except Exception as e:
            print(f"Error: {e}")
            continue
        
        llava_results_baseline_original_label[k] = webqa_accuracy(baseline_answer, example['A'], example['Qcate'].lower())        
        llava_results_blank_original_label[k] = webqa_accuracy(blank_answer, example['A'], example['Qcate'].lower())

        
        with open("results/answers.csv", "a") as f:
            f.write(f"{model_path},{k},baseline,{baseline_answer}\n")
            f.write(f"{model_path},{k},blank,{blank_answer}\n")
                    
        llava_results_perturbed_perturbed_label[k] = {}
        llava_results_perturbed_original_label[k] = {}
        llava_results_baseline_perturbed_label[k] = {}
        llava_results_blank_perturbed_label[k] = {}
        for idx, label in example['A_perturbed'].items():
            generated_image_files = []
            for img in example['img_posFacts']:
                if use_split:
                    generated_file = f"{perturbation_path}/{example['split']}/{str(img['image_id'])}_{k}_{idx}.jpeg"
                else:
                    generated_file = f"{perturbation_path}/{str(img['image_id'])}_{k}_{idx}.jpeg"
                if os.path.exists(generated_file):
                    generated_image_files.append(generated_file)
                else:
                    generated_image_files.append(str(img['image_id']))
            try:
                perturbed_answer = eval_on_webqa_sample(generated_image_files, example, processor, model, conversational_prompt)
            except Exception as e:
                print(f"Error: {e}")
                continue
            
            llava_results_perturbed_original_label[k][idx] = webqa_accuracy(perturbed_answer, example['A'], example['Qcate'].lower())
            llava_results_perturbed_perturbed_label[k][idx] = webqa_accuracy(perturbed_answer, [label], example['Qcate'].lower())
            llava_results_baseline_perturbed_label[k][idx] = webqa_accuracy(baseline_answer, [label], example['Qcate'].lower())
            llava_results_blank_perturbed_label[k][idx] = webqa_accuracy(blank_answer, [label], example['Qcate'].lower())
            
            with open("results/answers.csv", "a") as f:
                f.write(f"{model_path},{k},{str(idx)},{perturbed_answer}\n")
               
    results[model_path] = {
        "baseline_original_label": accuracy_agg_results(llava_results_baseline_original_label, eval_data),
        "baseline_perturbed_label": accuracy_agg_generated_results(llava_results_baseline_perturbed_label, eval_data),
        "blank_original_label": accuracy_agg_results(llava_results_blank_original_label, eval_data),
        "blank_perturbed_label": accuracy_agg_generated_results(llava_results_blank_perturbed_label, eval_data),
        "perturbed_original_label": accuracy_agg_generated_results(llava_results_perturbed_original_label, eval_data),
        "perturbed_perturbed_label": accuracy_agg_generated_results(llava_results_perturbed_perturbed_label, eval_data)
    }
    
    # dump to temp.json
    with open("temp.json", "w") as f:
        json.dump(results, f)
    # print(results[model_path])

formatted = {}
for model_path, result in results.items():
    formatted[model_path + '_1_image'] = {k:v for (k, (v, _, _)) in result.items()}
    formatted[model_path + '_2_image'] = {k:v for (k, (_, v, _)) in result.items()}
    formatted[model_path + '_N_image'] = {k:v for (k, (_, _, v)) in result.items()}
    # print(f"Model: {model_path}")
    # print(result)


formatted_df = pd.DataFrame(formatted).T
exp_name = __file__.split('/')[-1].split('.')[0]

if not save:
    print(formatted_df)
elif not os.path.exists(f"results/{exp_name}.csv"):
    formatted_df.to_csv(f"results/{exp_name}.csv")
else:
    formatted_df.to_csv(f"results/{exp_name}.csv", mode='a', header=False)