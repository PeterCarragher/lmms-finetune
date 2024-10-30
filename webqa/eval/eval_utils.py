from eval_1022 import *

import torch
from PIL import Image
import convert_webqa_data
from PIL import Image
import torch
from transformers import AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoModelForCausalLM 
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

def get_model_processor(model_path):
    if "llava-v1.6" in model_path or "llava-1.6" in model_path:
        processor = LlavaNextProcessor.from_pretrained(model_path)
        model = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")#, low_cpu_mem_usage=True) 
    elif "llava" in model_path:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path, # model_id, 
            torch_dtype=torch.float16, 
            # low_cpu_mem_usage=True, 
        ).to("cuda")
        processor = AutoProcessor.from_pretrained(model_path)
    elif "Qwen" in model_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_path)
    elif "Phi" in model_path:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True, torch_dtype="auto", _attn_implementation='flash_attention_2') 
        # use _attn_implementation='eager' to disable flash attention
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True) 
    else:
        raise ValueError(f"Unknown model path: {model_path}")
    return model, processor

def get_messages(data, conversational_prompt=True, reverse_images = False):#, img_token = "<image>"): # "<image-placeholder">
    imgs = data['img_posFacts'] if not reverse_images else data['img_posFacts'][::-1]
    if not conversational_prompt:
        prompt = ""
        for id, img in enumerate(imgs):
            prompt += f"<|image_{id + 1}|>\nCaption: {img['title']}\n"
        prompt += f"Q: {data['Q']}"
        return [{"role": "user", "content": prompt}]
    
    message = {"role": "user", "content": []}
    for img in imgs:
        message["content"].append({"type": "image"})
        message["content"].append({"type": "text", "text": f"Caption: {img['title']}"})
    
    message["content"].append({"type": "text", "text": f"Q: {data['Q']}"})
    return [message]

def get_images(image_paths, reverse_images = False):
    images = []
    if reverse_images:
        image_paths = image_paths[::-1]
        
    for image_path in image_paths:
        try:
            image_path_int = int(image_path)
        except:
            image_path_int = None
        
        if image_path_int:
            images.append(load_webqa_image(image_path))
        else:
            images.append(Image.open(image_path).convert("RGB"))

    return images

system_prompt = "Answer question Q based only on the provided images.\n"

def run_inference(messages, images, processor, model, conversational_prompt):
    if conversational_prompt:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=images, text=text, return_tensors='pt', padding=True)
    else:
        text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text, images, return_tensors="pt")
    inputs = inputs.to("cuda")
    output = model.generate(
        **inputs, 
        max_new_tokens=50,
        do_sample=False, 
        # max_length=100,
        # num_return_sequences=1,
        # temperature=0.0,
    )
    return processor.decode(output[0][2:], skip_special_tokens=True)


def eval_on_webqa_sample(image_paths, data, processor, model, conversational_prompt, reverse_images = False):
    images = get_images(image_paths, reverse_images)
    messages = get_messages(data, conversational_prompt, reverse_images)
    return run_inference(messages, images, processor, model, conversational_prompt)
    # query = f"SYSTEM: {system_prompt}\nHUMAN: {query}\nGPT:"
    # print(query) 


def webqa_accuracy(answer, label, Qcate):
    if Qcate == 'color':
        F1_avg, F1_max, EM, RE_avg, PR_avg = compute_vqa_metrics([answer], label[0], "", color_set)
    elif Qcate == 'shape': 
        F1_avg, F1_max, EM, RE_avg, PR_avg = compute_vqa_metrics([answer], label[0], "", shape_set)
    elif Qcate == 'yesno': 
        F1_avg, F1_max, EM, RE_avg, PR_avg = compute_vqa_metrics([answer], label[0], "", yesno_set)
    elif Qcate == 'number': 
        F1_avg, F1_max, EM, RE_avg, PR_avg = compute_vqa_metrics([answer], label[0], "", {"NUMBER"})
    else:
        return None
    return (F1_avg, F1_max, EM, RE_avg, PR_avg)

def ans_contains_any_label(ans, labels = ['yes', 'no']):
        return any([label in ans.lower() for label in labels])
    
def ans_contains_correct_label(ans, correct_ans, qcate):
    _,_,_,_,pr = webqa_accuracy(ans, correct_ans, qcate)
    return pr

def accuracy_agg_results(qa_results, eval_data):
    single_image_keys = [k for k in qa_results.keys() if len(eval_data[k]['img_posFacts']) == 1]
    two_image_keys = [k for k in qa_results.keys() if len(eval_data[k]['img_posFacts']) == 2]

    single_acc = np.mean([PR_avg for key, (F1_avg, F1_max, EM, RE_avg, PR_avg) in qa_results.items() if key in single_image_keys])
    two_image_acc = np.mean([PR_avg for key, (F1_avg, F1_max, EM, RE_avg, PR_avg) in qa_results.items() if key in two_image_keys])
    avr_acc = np.mean([PR_avg for key, (F1_avg, F1_max, EM, RE_avg, PR_avg) in qa_results.items()])
    return (single_acc, two_image_acc, avr_acc)

def accuracy_agg_generated_results(qa_results, eval_data):
    single_image_keys = [k for k in qa_results.keys() if len(eval_data[k]['img_posFacts']) == 1]
    two_image_keys = [k for k in qa_results.keys() if len(eval_data[k]['img_posFacts']) == 2]

    single_acc = np.mean([PR_avg for key, dict in qa_results.items() if key in single_image_keys for idx, (_,_,_,_,PR_avg) in dict.items()])
    two_image_acc = np.mean([PR_avg for key, dict in qa_results.items() if key in two_image_keys for idx, (_,_,_,_,PR_avg) in dict.items()])
    avr_acc = np.mean([PR_avg for key, dict in qa_results.items() for idx, (_,_,_,_,PR_avg) in dict.items()])
    
    return (single_acc, two_image_acc, avr_acc)