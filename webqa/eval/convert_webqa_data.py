import json
import numpy as np
    
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

def get_prompt(data, reverse_images = False, img_token = "<image>"): # "<image-placeholder">
    imgs = data['img_posFacts']
    if len(imgs) == 1:
        return f"{img_token} \n Caption: {imgs[0]['title']} \n Question: {data['Q']}"
    assert(len(imgs) == 2)
    if reverse_images:
        return f"{img_token} \n Caption: {imgs[1]['title']} \n {img_token} \n Caption: {imgs[0]['title']} \n Question: {data['Q']}"
    return f"{img_token} \n Caption: {imgs[0]['title']} \n {img_token} \n Caption: {imgs[1]['title']} \n Question: {data['Q']}"

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
                "system_prompt": "Answer the following questions by considering the given images.",
                "image": image_paths,
                "conversations": conversations
            }
            
            if entry['split'] == 'train':
                train_output.append(output_entry)
            else:
                val_output.append(output_entry)
    return train_output, val_output

def convert_perturbed_data(perturbed_data):
    generated_samples = {}
    for key in perturbed_data:
        entry = perturbed_data[key]
        img_id = entry['img_posFacts'][0]['image_id']
        for sample_id, label in entry['A_perturbed'].items():
            generated_image_path = f"/home/pcarragh/dev/webqa/segment/Inpaint-Anything/results/webqa/{entry['split']}/{str(img_id)}_{key}_{sample_id}.jpeg"
            entry['img_posFacts'][0]['image_id'] = generated_image_path
            entry['A'] = [label]
            generated_samples[key + '_' + str(img_id) + '_' + str(sample_id)] = entry
    return convert_format(generated_samples)

if __name__ == "__main__":
    data = json.load(open("/home/pcarragh/dev/webqa/MultiModalQA/data/WebQA_train_val_obj_v2.json.pert.lama.v3", "r"))
    perturbed_data = json.load(open("/home/pcarragh/dev/webqa/LLaVA/WebQA_train_val_color_gpt_matched.json", "r"))
    train_output, val_output = convert_format(data)
    print(len(train_output), len(val_output))

    gen_train_output, gen_val_output = convert_perturbed_data(perturbed_data)
    print(len(gen_train_output), len(gen_val_output))

    train_output.extend(gen_train_output)
    val_output.extend(gen_val_output)
    print(len(train_output), len(val_output))

    # Save the result to a new JSON file
    with open('webqa_train_gen_formatted.json', 'w') as f:
        json.dump(train_output, f, indent=4)
        
    with open('webqa_val_gen_formatted.json', 'w') as f:
        json.dump(val_output, f, indent=4)

    print("Conversion complete. Output saved.")
