import json
import pandas as pd
import random
import os
from tqdm import tqdm
from glob import glob
from collections import defaultdict, Counter
# from .glossary import normalize_word
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from io import BytesIO
import base64
import string, re
import spacy
from tasks.webqa.webqa_segmentation_task import webqa_category_labels
from tasks.webqa.arrow_format import domain_dict, toNum, normalize_text, find_first_search_term
import numpy as np
import time
from openai import OpenAI
import requests
import base64
import sys
nlp = spacy.load("en_core_web_sm", disable=["ner","textcat","parser"])

def is_valid(url):
    # only accept 'png', 'jpeg', 'gif', 'webp'
    if url[-4:] in ['.png', 'jpeg', '.gif', 'webp', '.jpg']:
        # check if the URL is live
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
            }
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                return False
        except:
            return False
        return True
    return False

def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
        
def format_query(query, img_path):
    query_list = [
        {"type": "text", "text": query['Q']},
    ]
    
    if img_path:
        file_type = img_path[-4:] == '.jpeg' and 'jpeg' or 'png'
        image_bytes = encode_image(img_path)
        query_list.append({"type": "text", "text": f"Caption: {query['img_posFacts'][0]['title']}"})
        query_list.append({"type": "image_url", "image_url": {"url": f"data:image/{file_type};base64,{image_bytes}"}})#, "detail": "low"}})
    
    for img in query['img_posFacts'][1:]:
        if is_valid(img['imgUrl']):
            query_list.append({"type": "text", "text": f"Caption: {img['title']}"})
            query_list.append({"type": "image_url", "image_url": {"url": f"{img['imgUrl']}", "detail": "low"}})
    return query_list

def get_response(query, client, model="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Answer the question in one word using the following image caption pairs. You must use the provided image sources to answer the question. If the answer is not in the image, respond 'unknown'."},
            {"role": "user", "content": query},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content

client = OpenAI(api_key="<KEY>")

def get_gpt_label_for_generated_sample(perturbed_idx, perturbed_answer, content, generated_image_path):
    generated_image_file = generated_image_path + str(perturbed_idx) + ".jpeg"
    perturbed_answer = normalize_text(perturbed_answer)
    perturbed_answer = find_first_search_term(perturbed_answer, webqa_category_labels[qcate], qcate, perturbed_answer)

    if not os.path.exists(generated_image_file):
        return None
    
    try:
        query = format_query(content, generated_image_file)
        gpt_answer = get_response(query, client)
    except Exception as e:
        gpt_answer = str(e)
    
    gpt_answer = normalize_text(gpt_answer)
    return gpt_answer

if __name__ == "__main__":
    dataset_path = "/home/pcarragh/dev/webqa/MultiModalQA/data/WebQA_train_val_obj_v2.json.pert.lama.v3.val"
    perturbed_dataset = json.load(open(dataset_path, "r"))
    for idx, k in enumerate(list(perturbed_dataset.keys())):
        content = perturbed_dataset[k]
        qcate = content['Qcate'].lower()
        if not qcate in webqa_category_labels.keys():
            continue
        
        img_posFacts = content['img_posFacts']
        
        if not len(img_posFacts) in [1,2]:
            continue
        
        generated_img_id = img_posFacts[0]['image_id']
        generated_image_path = f"/home/pcarragh/dev/webqa/segment/Inpaint-Anything/results/webqa/{content['split']}/{str(generated_img_id)}_{k}_"
        generated_image_file = generated_image_path + "0.jpeg"
        perturbed_dataset[k]['A_perturbed_gpt'] = {}

        if not (qcate in ['color', 'shape'] and 'A_perturbed' in content and os.path.exists(generated_image_file)):
            continue
        
        orig_answer = content['A'][0]
        orig_answer = normalize_text(orig_answer)
        orig_answer = find_first_search_term(orig_answer, domain_dict[qcate], qcate, orig_answer)
        if not orig_answer:
            print("Error: ", content['Guid'])
            orig_answer = content['A'][0]
            print(orig_answer)
        
        unmatched_idxs = []
        for perturbed_idx, perturbed_answer in content['A_perturbed'].items():
            gpt_answer = get_gpt_label_for_generated_sample(perturbed_idx, perturbed_answer, content, generated_image_path)
            if not gpt_answer:
                continue
            perturbed_dataset[k]['A_perturbed_gpt'][perturbed_idx] = gpt_answer
            print(f"{round(100*idx/len(perturbed_dataset), 2)}, GPT: {gpt_answer}, Perturbed: {perturbed_answer}, Original: {orig_answer}. QID: {content['Guid']},  Q: {content['Q']}")
            time.sleep(3)
            with open(dataset_path + '.gpt.csv', "a") as f:
                output = ','.join([str(content['Guid']), str(perturbed_idx), orig_answer, perturbed_answer, gpt_answer])
                f.write(output + '\n')
            
            if orig_answer[:3] != gpt_answer[:3]:
                unmatched_idxs.append(perturbed_idx)
                
        for idx in unmatched_idxs:
            del perturbed_dataset[k]['A_perturbed'][idx]
            if idx in perturbed_dataset[k]['A_perturbed_gpt']:
                del perturbed_dataset[k]['A_perturbed_gpt'][idx]
    
    with open(dataset_path + '.gpt_filtered.json', "w") as f:
        json.dump(perturbed_dataset, f)