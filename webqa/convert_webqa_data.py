import json
import numpy as np

def get_prompt(data, key):
    imgs = data[key]['img_posFacts']
    if len(imgs) == 1:
        return f"<image> \n Caption: {imgs[0]['title']} \n Question: {data[key]['Q']}"
    assert(len(imgs) == 2)
    return f"<image> \n Caption: {imgs[0]['title']} \n <image> \n Caption: {imgs[1]['title']} \n Question: {data[key]['Q']}"

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
            prompt = get_prompt(data, key)
            conversations = [
                {
                    "from": "human",
                    "value": prompt
                },
                {
                    "from": "gpt",
                    "value": entry['A'][0]
                }
            ]
            
            image_paths = []
            for img in entry['img_posFacts']:
                image_paths.append(img['image_id'])
            
            output_entry = {
                "system_prompt": "Answer the following questions by considering the given images.",
                "image": image_paths,
                "conversations": conversations
            }
            
            if entry['split'] == 'train':
                train_output.append(output_entry)
            else:
                val_output.append(output_entry)
    return train_output, val_output


data = json.load(open("/home/pcarragh/dev/webqa/MultiModalQA/data/WebQA_train_val_obj_v2.json.pert.lama.v3", "r"))
train_output, val_output = convert_format(data)

# Save the result to a new JSON file
with open('webqa_train_formatted.json', 'w') as f:
    json.dump(train_output, f, indent=4)
    
with open('webqa_val_formatted.json', 'w') as f:
    json.dump(val_output, f, indent=4)

print("Conversion complete. Output saved to 'webqa_formatted.json'.")
