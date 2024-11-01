import json
from tqdm import tqdm
import sys
sys.path.append("..")
from eval.eval_utils import *

# eval_data = json.load(open("/home/pcarragh/dev/webqa/MultiModalQA/data/WebQA_train_val_obj_v2.json.pert.lama.v3", "r"))
eval_data = json.load(open("/home/pcarragh/dev/webqa/MultiModalQA/data/WebQA_train_val_obj_v2.json", "r"))
perturbation_path = "/home/pcarragh/dev/webqa/segment/Inpaint-Anything/results/webqa"
version = "2"
model_path = "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4" 
model, processor = get_model_processor(model_path)
conversational_prompt = not 'Phi' in model_path
output_file = "qa_check_perturbations"
with open(f"data/{output_file}_v{version}.csv", "w") as f:
    f.write("model,file,qa_check\n")

for file in tqdm(os.listdir(perturbation_path)):
    if not file.endswith(".jpeg"):
        continue
    image_id = file.split("_")[0]
    key = file.split("_")[1]
    A_perturbed_idx = file.split("_")[2].split(".")[0]
    example = eval_data[key]
    counterfactual_image_file = f"{perturbation_path}/{file}"
    try:
        question = f"Q: is the {example['Qcate']} of the {example['Q_obj']} the same in both images?"
        messages = get_qa_check_prompt(question, conversational_prompt)
        images = get_images([image_id, counterfactual_image_file])
        qa_check = run_inference(messages, images, processor, model, conversational_prompt)
    except Exception as e:
        print(f"Error: {e}")
        continue
    qa_check = qa_check.split('\n')[-1]
    with open(f"data/{output_file}_v{version}.csv", "a") as f:
        f.write(f"{model_path},{file},\"{qa_check}\"\n")
