import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# export LD_LIBRARY_PATH=/home/pcarragh/miniconda3/envs/lmms-finetune/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
import json
from tqdm import tqdm
import sys
sys.path.append("..")
from eval.eval_utils import *

eval_data = json.load(open("/home/pcarragh/dev/webqa/MultiModalQA/data/WebQA_train_val_obj_v2.json", "r"))
# eval_data = {k: v for k, v in eval_data.items() if v['Qcate'].lower() == 'yesno'}
# perturbation_path = "/home/pcarragh/dev/webqa/image_gen_val/val_images_perturbed_gpt_obj_lama"
perturbation_path = "/home/pcarragh/dev/webqa/image_gen_val/val_images_perturbed_gpt_obj_lama"
version = "2"
# model_path = "Qwen/Qwen2-VL-72B-Instruct-AWQ" #
# model_path = "Qwen/Qwen2-VL-7B-Instruct" # 
model_path = "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4"
model, processor = get_model_processor(model_path)
conversational_prompt = not 'Phi' in model_path
output_file = "qa_check_counterfactuals"
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
    try:
        images = get_images([image_id, counterfactual_image_file])
        qa_check = run_inference(messages, images, processor, model, conversational_prompt)
    except Exception as e:
        # print(f"Error: {e}")
        continue
    qa_check = qa_check.split('\n')[-1]
    with open(f"data/{output_file}_v{version}.csv", "a") as f:
        f.write(f"{model_path},{file},\"{qa_check}\"\n")

