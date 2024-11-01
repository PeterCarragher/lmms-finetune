import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# export LD_LIBRARY_PATH=/home/pcarragh/miniconda3/envs/lmms-finetune/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
import json
from tqdm import tqdm
import sys
sys.path.append("..")
from eval.eval_utils import *
import gc

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4")
    parser.add_argument("--version", type=str, default="2")
    parser.add_argument("--perturbation_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="qa_check_counterfactuals")
    parser.add_argument("--eval_data_path", type=str, required=True)
    args = parser.parse_args()
    # model_path = "Qwen/Qwen2-VL-72B-Instruct-AWQ" #
    # model_path = "Qwen/Qwen2-VL-7B-Instruct" # 
    # model_path = "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4"
    model_path = args.model_path
    version = args.version
    perturbation_path = args.perturbation_path
    output_file = args.output_file
    eval_data = json.load(open(args.eval_data_path, "r"))

    model, processor = get_model_processor(model_path)
    conversational_prompt = not 'Phi' in model_path
    output_file = __file__.split("/")[-1].replace(".py", "")
    with open(f"data/{output_file}_v{version}.csv", "w") as f:
        f.write("model,file,qa_check\n")

    for file in tqdm(os.listdir(perturbation_path)):
        if not file.endswith(".jpeg"):
            continue
        image_id = file.split("_")[0]
        key = file.split("_")[1].split(".")[0]
        example = eval_data[key]
        counterfactual_image_file = f"{perturbation_path}/{file}"
        question = f"Q: is the {example['Q_obj']} in both the original image and the perturbed image?"
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
        with open(f"data/{output_file}_v{version}.csv", "a") as f:
            f.write(f"{model_path},{file},\"{qa_check}\"\n")

