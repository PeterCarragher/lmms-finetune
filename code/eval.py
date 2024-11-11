import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append(".")
from eval_utils import *
import random
from copy import deepcopy

random.seed(42)
dataset = json.load(open("../data/segsub_data_val_v4.json", "r"))
segsub_dir = "/data/nikitha/VQA_data/segsub_images"
coco_image_dir = "/data/nikitha/VQA_data/VQAv2/images/val2014"
# webqa_image_dir = "/data/nikitha/VQA_data/images/webqa/images"

def get_image_paths(sample):
    paths = []
    for image in sample['image']:
        if isinstance(image, int):
            assert(sample['dataset'] == 'webqa')
            paths.append(image)
        elif sample['type'] == 'original':
            assert(sample['dataset'] in ['vqa', 'okvqa'])
            paths.append(f"{coco_image_dir}/{image}")
        else:
            paths.append(f"{segsub_dir}/{image}")
    return paths

original_samples = {}
for idx, sample in enumerate(dataset):
    if sample['type'] == 'original':
        sample['answer_idx'] = idx
        original_samples[f"{sample['dataset']}_{sample['id']}"] = sample

# old models
# ("/home/nikithar/Code/VQA/lmms-finetune/checkpoints/llava-1.5-7b_v2_lora-True_qlora-False/", "llava-hf/llava-1.5-7b-hf"), # RET trained

model_paths = [
    "llava-hf/llava-1.5-7b-hf",
    ("/home/nikithar/Code/VQA/lmms-finetune/checkpoints/data_v2/llava-1.5-7b_v2_lora-True_qlora-False", "llava-hf/llava-1.5-7b-hf"),
    "llava-hf/llava-1.5-13b-hf",
    "Qwen/Qwen2-VL-7B-Instruct",
    "microsoft/Phi-3-vision-128k-instruct",
    # ("../finetuned_results/qwen2_finetuned_v1.jsonl", "Qwen/Qwen2-VL-7B-Instruct"),
    # ("../finetuned_results/qwen2_finetuned_v2.jsonl", "Qwen/Qwen2-VL-7B-Instruct"),
    # ("../finetuned_results/phi3_finetuned_v1.jsonl", "microsoft/Phi-3-vision-128k-instruct"),
    # "../finetuned_results/gpt-4o-mini.jsonl",
]

def eval_metrics(old_answer, new_answer, old_label, new_label, qcate=None, blank_answer=None): 
    res = {
        "img_old,label_old": ans_contains_correct_label(old_answer, old_label, qcate),
        "img_new,label_old": ans_contains_correct_label(new_answer, old_label, qcate),
        "img_old,label_new": ans_contains_correct_label(old_answer, new_label, qcate),
        "img_new,label_new": ans_contains_correct_label(new_answer, new_label, qcate),
        "img_old,label_ret": retrieval_predicted(old_answer),
        "img_new,label_ret": retrieval_predicted(new_answer),
    }
    if qcate and isinstance(qcate, list):
        res["img_old,label_any"] = ans_contains_any_label(old_answer, qcate)
        res["img_new,label_any"] = ans_contains_any_label(new_answer, qcate)
    if blank_answer:
        res["img_blank,label_old"] = ans_contains_correct_label(blank_answer, old_label, qcate)
        res["img_blank,label_new"] = ans_contains_correct_label(blank_answer, new_label, qcate) 
        res["img_blank,label_ret"] = retrieval_predicted(blank_answer)
    return res

results_config = {
        "webqa": {
            "perturbed": [],
            "conflicting": [],
            "counterfactual": [],
        },
        "vqa": {            
            # "perturbed": [],
            # "conflicting": [],
            "counterfactual": [],
        },
        "okvqa": {
            # "perturbed": [],
            # "conflicting": [],
            "counterfactual": [],
        },
    }

results = {}
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=int, default=None)
    parser.add_argument("--save", type=bool, default=False)
    args = parser.parse_args()
    save = args.save
    model_id = args.model_id
    if model_id:
        model_paths = [model_paths[model_id]]

    for model_path in model_paths:  
        if isinstance(model_path, tuple):
            model_path, original_model_id = model_path
        else:
            original_model_id = model_path
        
        if os.path.exists(f"../results/models/{model_path.split('/')[-1]}_results.json"):
            results[model_path] = json.load(open(f"../results/models/{model_path.split('/')[-1]}_results.json", "r"))
            print(f"Results already exist for model: {model_path}")
            continue    

        print(f"Running evaluation for model: {model_path}")
        
        conversational_prompt = not 'Phi' in model_path
        
        if model_path.endswith(".jsonl"):
            print(f"Reading answers from {model_path}")
            model = None
            processor = None
            answers = []
            with open(model_path, "r") as f:
                for line in f:
                    answers.append(json.loads(line))
        else:
            model, processor = get_model_processor(model_path, original_model_id)
        
        results[model_path] = deepcopy(results_config)
        
        # sample dataset
        # dataset = random.sample(dataset, 500)
        for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
            eval_dataset_id = sample['dataset']
            eval_type = sample['type']
            if eval_type == 'original' or not f"{eval_dataset_id}_{sample['id']}" in original_samples:
                continue
            original_sample = original_samples[f"{eval_dataset_id}_{sample['id']}"]
            old_label = original_sample['conversations'][1]["value"]

            new_label = sample['conversations'][1]["value"]
            if model:
                input_text = sample['conversations'][0]["value"]
                question = input_text.split("Q: ")[-1]
                captions = input_text.split("Q: ")[0].split("<image>\nCaption: ")
                if len(captions) > 1:
                    captions = captions[1:]
                    captions = [caption.split("\n")[0] for caption in captions]
                else:
                    captions = []
                try:
                    new_images = get_image_paths(sample)
                    new_answer = eval_on_sample(new_images, question, captions, processor, model, conversational_prompt)
                    old_images = get_image_paths(original_sample)
                    old_answer = eval_on_sample(old_images, question, captions, processor, model, conversational_prompt)
                except Exception as e:
                    print(f"Error: {e}")
                    continue
            else:
                new_answer = answers[idx]['response']
                old_sample_idx = original_sample['answer_idx']
                old_answer = answers[old_sample_idx]['response']
        
            # TODO: blank image variants
            qcate = None
            if eval_dataset_id in ['vqa', 'okvqa']:
                qcate = eval_dataset_id
            elif eval_type == 'perturbed':
                qcate = ['color', 'shape']
            elif eval_type == 'conflicting':
                qcate = ['color', 'shape']
            elif eval_type == 'counterfactual':
                qcate = ['yesno']
            
            # TODO: Blank metrics
            results[model_path][eval_dataset_id][eval_type].append(eval_metrics(old_answer, new_answer, old_label, new_label, qcate))

        # Save results
        with open(f"../results/models/{model_path.split('/')[-1]}_results.json", "w") as f:
            json.dump(results[model_path], f)

    # aggregate and save all results
    if save:
        all_results = []
        for eval_dataset_id in ['webqa', 'vqa', 'okvqa']:
            for eval_type in ['perturbed', 'conflicting', 'counterfactual']:
                if eval_type not in results_config[eval_dataset_id]:
                    continue
                results_agg = {}
                for model_path in model_paths:
                    if isinstance(model_path, tuple):
                        model_path, _ = model_path
                    results_agg[model_path] = {
                        "img_old,label_old": [],
                        "img_new,label_old": [],
                        "img_old,label_new": [],
                        "img_new,label_new": [],
                        "img_old,label_ret": [],
                        "img_new,label_ret": [],
                    }
                    
                    for res in results[model_path][eval_dataset_id][eval_type]:
                        for k, v in res.items():
                            if k in results_agg[model_path]:
                                results_agg[model_path][k].append(v)
                    results_agg[model_path] = {k: np.mean(v) for k, v in results_agg[model_path].items()}

                # Save individual CSV files
                df = pd.DataFrame(results_agg).T
                df.to_csv(f"../results/{eval_dataset_id}_{eval_type}_results.csv")

                # Add dataset and type columns, append to all_results
                df['dataset'] = eval_dataset_id
                df['type'] = eval_type
                all_results.append(df.reset_index(names=["model_path"],inplace=False))

        # Concatenate all results and save to a single CSV
        all_results_df = pd.concat(all_results, ignore_index=True)
        all_results_df.to_csv("../results/all_results.csv", index=False)