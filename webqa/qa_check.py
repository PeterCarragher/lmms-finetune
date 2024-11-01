import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# export LD_LIBRARY_PATH=/home/pcarragh/miniconda3/envs/lmms-finetune/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
import json
from tqdm import tqdm
import sys
sys.path.append("..")
from eval.eval_utils import *
import gc
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4")
    parser.add_argument("--version", type=str, default="2")
    parser.add_argument("--perturbation_path", type=str, required=True)
    parser.add_argument("--eval_data_path", type=str, required=True)
    parser.add_argument("--counterfactual", type=bool, default=True)
    args = parser.parse_args()
    # model_path = "Qwen/Qwen2-VL-72B-Instruct-AWQ" #
    # model_path = "Qwen/Qwen2-VL-7B-Instruct" # 
    # model_path = "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4"
    model_path = args.model_path
    version = args.version
    perturbation_path = args.perturbation_path
    eval_data = json.load(open(args.eval_data_path, "r"))
    use_counterfactual_question = args.counterfactual

    model, processor = get_model_processor(model_path)
    conversational_prompt = not 'Phi' in model_path
    output_str = "counterfactual" if use_counterfactual_question else "perturbation"
    output_file = f"data/qa_checkk_{output_str}_v{version}.csv"
    
    with open(output_file, "w") as f:
        f.write("model,file,qa_check\n")

    for file in tqdm(os.listdir(perturbation_path)):
        if not file.endswith(".jpeg"):
            continue
        image_id = file.split("_")[0]
        key = file.split("_")[1].split(".")[0]
        example = eval_data[key]
        counterfactual_image_file = f"{perturbation_path}/{file}"
        if use_counterfactual_question:
            question = f"Q: is the {example['Q_obj']} in both the original image and the perturbed image?"
        else:
            question = f"Q: is the {example['Qcate']} of the {example['Q_obj']} the same in both images?"

        messages = get_qa_check_prompt(question, conversational_prompt)
        images = None
        try:
            images = get_images([image_id, counterfactual_image_file])
            qa_check = run_inference(messages, images, processor, model, conversational_prompt)
        except Exception as e:
            # print(f"Error: {e}")
            torch.cuda.empty_cache()
            del messages
            if images:
                del images
            gc.collect()
            continue
        qa_check = qa_check.split('\n')[-1]
        with open(output_file, "a") as f:
            f.write(f"{model_path},{file},\"{qa_check}\"\n")

