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
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

# import kornia
from transformers import set_seed
from vcd_utils.vcd_add_noise import add_diffusion_noise
from vcd_utils.vcd_sample import evolve_vcd_sampling
evolve_vcd_sampling()

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
 
    with open(args.question_file, "r", encoding="utf-8") as f:
        queries = json.load(f)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for line in tqdm(queries):
        id = line["id"]
        query = line["query"]
        image= line["image"]
        
        if model.config.mm_use_im_start_end:
            query = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + query
        else:
            query = DEFAULT_IMAGE_TOKEN + '\n' + query

        conv = conv_templates[args.conv_mode].copy()
        if int(id) > 1004:
            conv.append_message(conv.roles[0], query + " Please answer this question with one word.")
        else:
            conv.append_message(conv.roles[0], query)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        image = Image.open(os.path.join(args.image_folder, image))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        if args.use_cd:
            if args.use_zcd:
                image_tensor_cd = torch.full_like(image_tensor, 0) ##zcd 추가함
            else:
                image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step) ##vcd 부분
        else:
            image_tensor_cd = None      

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                images_cd=(image_tensor_cd.unsqueeze(0).half().cuda() if image_tensor_cd is not None else None),
                cd_alpha = args.cd_alpha,
                cd_beta = args.cd_beta,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=1024,
                use_cache=True,
                use_flb = args.use_flb,
                flb_gamma = args.flb_gamma,
                flb_lambda = args.flb_lambda,
                )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
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
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--model-path", type=str, default="./checkpoints/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/home/de-hallucination/ZCD/AMBER/image")
    parser.add_argument("--question-file", type=str, default="/home/de-hallucination/ZCD/AMBER/data/query/query_generative.json")
    parser.add_argument("--answers-file", type=str, default="output_pruning/SENTENCE.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None) ## greedy when 1

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
