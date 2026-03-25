# This file is part of the VCD project: https://github.com/DAMO-NLP-SG/VCD
# Original work: (c) the authors of VCD, licensed under the Apache License 2.0.
#
# Modifications:
#   - 2025-12-01: Added benchmark setting
#
# Modified by: Jiwoo Ha (DGIST Distributed AI Lab)
# (c) 2025 Jiwoo Ha. All rights reserved for the modifications only.

import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
from transformers import set_seed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
import math

import kornia
from lavis.models import load_model_and_preprocess
from vcd_utils.vcd_add_noise import add_diffusion_noise
from vcd_utils.vcd_sample import evolve_vcd_sampling
evolve_vcd_sampling()

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # loads InstructBLIP model
    # For large_sized model,
    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)

    with open(args.question_file, "r", encoding="utf-8") as f:
        queries = json.load(f)
    
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    for line in tqdm(queries):
        id = line["id"]
        query = line["query"]
        image= line["image"]

        if int(id) > 1004:
            prompt = query +  " Please answer this question with one word."
        else:
            prompt = query

        raw_image = Image.open(os.path.join(args.image_folder, image)).convert("RGB")
        # prepare the image
        image_tensor = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        ## create a white image for contrastive decoding
        if args.use_cd:
            if args.use_zcd:
                image_tensor_cd = torch.full_like(image_tensor, 0) ##zcd 추가함
            else:
                image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step) ##vcd 부분
        else:
            image_tensor_cd = None      
        with torch.inference_mode():
            outputs = model.generate({"image": image_tensor, "prompt": prompt},
                use_nucleus_sampling=True, num_beams=1,
                top_p = args.top_p, repetition_penalty=1,
                images_cd=image_tensor_cd, cd_alpha = args.cd_alpha, cd_beta = args.cd_beta,
                use_flb = args.use_flb, flb_gamma = args.flb_gamma, flb_lambda = args.flb_lambda
                )

        outputs = outputs[0]
        ans_file.write(json.dumps({"id": id,
                                   "response": outputs
                                   }, ensure_ascii=False)+"\n")
        ans_file.flush()
    ans_file.close()

    ## jsonl to json
    json_list = []

    with open(answers_file, "r", encoding="utf-8") as infile:
        for line in infile:
            json_obj = json.loads(line.strip())
            json_list.append(json_obj)

    with open(answers_file[:len(answers_file)-1], "w", encoding="utf-8") as outfile:
        json.dump(json_list, outfile, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    def str2bool(v):
        return v.lower() in ("true", "1", "yes")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")

    # model 
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None) 

    # vcd
    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_cd", type=str2bool, default=False)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)

    # zcd
    parser.add_argument("--use_zcd", type=str2bool, default=False)
    
    # FLB
    parser.add_argument("--use_flb", type=str2bool, default=False)
    parser.add_argument("--flb_gamma", type=float, default=0.3)
    parser.add_argument("--flb_lambda", type=float, default=0.05)

    args = parser.parse_args()
    print(args)
    set_seed(args.seed)
    eval_model(args)
