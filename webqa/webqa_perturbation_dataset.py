import json
import numpy as np
import pandas as pd

def get_conversation(entry, reverse_images = False, supervised = True):
    prompt = get_prompt(entry, reverse_images)
    conversations = [
        {
            "from": "human",
            "value": prompt
        },
    ]
    if supervised:
        conversations.append(
            {
                "from": "gpt",
                "value": entry['A'][0]
            }
        )
    return conversations

def get_prompt(data, reverse_images = False, img_token = "<image>"):
    imgs = data['img_posFacts'] if not reverse_images else data['img_posFacts'][::-1]
    prompt = ""
    for _, img in enumerate(imgs):
        prompt += f"{img_token}\nCaption: {img['title']}\n"
    prompt += f"Q: {data['Q']}"
    return prompt
    

# def get_image_path(image_id):
#     # TODO: extract webqa images first
#     # return f"{img['image_id']}.jpeg")
#     pass

def convert_format(data):
    train_output = []
    val_output = []
    for key in data:
        entry = data[key]
        if 'img_posFacts' in entry and len(entry['img_posFacts']) > 0:
            conversations = get_conversation(entry)
            
            image_paths = []
            for img in entry['img_posFacts']:
                image_paths.append(img['image_id'])
            
            output_entry = {
                "system_prompt": "Answer the question Q. If you need help answer <RET> to get the context.",
                "image": image_paths,
                "conversations": conversations
            }
            
            if entry['split'] == 'train':
                train_output.append(output_entry)
            else:
                val_output.append(output_entry)
    return train_output, val_output

def is_counterfactual_sample(entry):
    return 'yes' in entry['A'][0].lower() and entry['Qcate'].lower()  == 'yesno'

def is_conflicting_sample(entry):
    return len(entry['img_posFacts']) == 2 and 'A_perturbed' in entry and entry['Qcate'] in ['color', 'shape']

def is_perturbed_sample(entry):
    return len(entry['img_posFacts']) == 1 and 'A_perturbed' in entry and entry['Qcate'] in ['color', 'shape']

def convert_perturbed_data(perturbed_data):
    generated_samples = {}
    for key in perturbed_data:
        entry = perturbed_data[key]
        img_id = entry['img_posFacts'][0]['image_id']
        
        if is_perturbed_sample(entry):
            for sample_id, label in entry['A_perturbed'].items():
                generated_image_path = f"/home/pcarragh/dev/webqa/segment/Inpaint-Anything/results/webqa/{entry['split']}/{str(img_id)}_{key}_{sample_id}.jpeg"
                entry['img_posFacts'][0]['image_id'] = generated_image_path
                entry['A'] = [label]
                generated_samples[key + '_' + str(img_id) + '_' + str(sample_id)] = entry
        elif is_conflicting_sample(entry):
            # TODO: with new generations, we can take every permutation of 1st and 2nd image, generated and ungenerated
            # also we will have 2 image generations where both images are generated to same label, these are not conflicting but are positive examples
            for sample_id, label in entry['A_perturbed'].items():
                generated_image_path = f"/home/pcarragh/dev/webqa/segment/Inpaint-Anything/results/webqa/{entry['split']}/{str(img_id)}_{key}_{sample_id}.jpeg"
                entry['img_posFacts'][0]['image_id'] = generated_image_path
                entry['A'] = ['<RET>']
                generated_samples[key + '_' + str(img_id) + '_' + str(sample_id)] = entry
        elif is_counterfactual_sample(entry):
            generated_image_path = f"/home/pcarragh/dev/webqa/image_gen_val/val_images_perturbed_gpt_obj_lama/{str(img_id)}_{key}.jpeg"
            entry['img_posFacts'][0]['image_id'] = generated_image_path
            entry['A'] = ['<RET>']
            generated_samples['countefactual_' + key + '_' + str(img_id)] = entry
    return convert_format(generated_samples)

if __name__ == "__main__":
    version = 2
    data = json.load(open("/home/pcarragh/dev/webqa/MultiModalQA/data/WebQA_train_val_obj_v2.json.pert.lama.v3", "r"))
    perturbed_data = json.load(open("/home/pcarragh/dev/webqa/LLaVA/WebQA_train_val_color_gpt_matched.json", "r"))
    counterfactual_data = json.load(open("/home/pcarragh/dev/webqa/MultiModalQA/data/WebQA_train_val_obj_v2.json", "r"))
    qids = list(set(pd.read_csv('data/counterfactual_qa_check.csv', header=None)[0].tolist()))
    counterfactual_data = {k: v for k, v in counterfactual_data.items() if k in qids and is_counterfactual_sample(v)}
        
    train_output, val_output = convert_format(data)
    print(len(train_output), len(val_output))

    perturbed_train_output, perturbed_val_output = convert_perturbed_data(perturbed_data)
    print(len(perturbed_data), len(perturbed_train_output), len(perturbed_val_output))
    train_output.extend(perturbed_train_output)
    val_output.extend(perturbed_val_output)

    counterfactual_train_output, counterfactual_val_output = convert_perturbed_data(counterfactual_data)
    print(len(counterfactual_data), len(counterfactual_train_output), len(counterfactual_val_output))
    train_output.extend(counterfactual_train_output)
    val_output.extend(counterfactual_val_output)
    print(len(train_output), len(val_output))

    # Save the result to a new JSON file
    with open(f'data/webqa_train_gen_formatted_v{version}.json', 'w') as f:
        json.dump(train_output, f, indent=4)
        
    with open(f'data/webqa_val_gen_formatted_v{version}.json', 'w') as f:
        json.dump(val_output, f, indent=4)

    print("Conversion complete. Output saved.")
