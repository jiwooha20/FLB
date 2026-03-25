# This file is part of the VCD project: https://github.com/DAMO-NLP-SG/VCD
# Original work: (c) the authors of VCD, licensed under the Apache License 2.0.
#
# Modifications:
#   - 2025-12-01: Added benchmark setting
#
# Modified by: Jiwoo Ha (DGIST Distributed AI Lab)
# (c) 2025 Jiwoo Ha. All rights reserved for the modifications only.

# for random seed, gpu id
seed=${1:-55}
gpu_id=${2:-1}

# for AMBER dataset type
AMBER_folder=${3:-"/home/de-hallucination/ZCD/AMBER"}
query_type=${4:-"query_generative"} 

cd_beta=${5:-0.1}
# for VCD parameters
use_cd=${6:-"false"}
cd_alpha=${7:-1}
noise_step=${8:-500} 

# for FLB parameters
use_flb=${9:-"true"} # do not set true both vcd and flb
flb_gamma=${10:-0.3}
flb_lambda=${11:-0.05}

# for answers file naming
output_folder=${12:-"/home/de-hallucination/ZCD/FLB/experiments/output/251201_test"}
mkdir -p "${output_folder}"

base_name="instructblip_AMBER_answers_seed${seed}"
if [[ "$use_cd" == "true" ]]; then suffix="_vcd_alpha${cd_alpha}_beta${cd_beta}"
else suffix=""
fi
if [[ "$use_flb" == "true" ]]; then suffix="_flb_beta${cd_beta}_gamma${flb_gamma}_lambda${flb_lambda}"
else suffix=""
fi
answers_file="${output_folder}/${base_name}${suffix}.jsonl"

CUDA_VISIBLE_DEVICES=${gpu_id} python ./eval/object_hallucination_vqa_instructblip_AMBER.py \
--question-file /home/de-hallucination/ZCD/AMBER/data/query/${query_type}.json \
--image-folder ${AMBER_folder}/image \
--answers-file ${answers_file} \
--use_cd $use_cd \
--cd_alpha $cd_alpha \
--cd_beta $cd_beta \
--noise_step $noise_step \
--use_flb $use_flb \
--flb_gamma ${flb_gamma} \
--flb_lambda ${flb_lambda} \
--seed ${seed} \