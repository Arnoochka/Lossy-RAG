from functools import partial

import bench
from utils import model_generate

#hf_transformers
#meta-llama/Llama-3.2-3B-Instruct ~ 6.5 Gb
#google/gemma-3-4b-pt ~ 8.6 Gb
#Qwen/Qwen3.5-4B ~ 8.4Gb
#HuggingFaceTB/SmolLM3-3B ~ 6.2Gb

#strata
#Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS
#Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS

#llama.cpp
#Qwen3.8-27B-UD-Q3_K_XL.gguf

model_name = "Qwen3.8-27B-UD-Q3_K_XL"

ifbench_json_path = f"src/bench/log/ifbench_{model_name}.jsonl"
diamond_json_path = f"src/bench/log/gpqa_diamond_{model_name}.jsonl"
aime_json_path = f"src/bench/log/aime_{model_name}.jsonl"

generate = partial(
    model_generate.opeanai_model_generate_single_answer,
    model_name=model_name,
    max_new_tokens=4096,
    reasoning_budget_tokens=3072,
    do_sample=True,
    temperature=1,
    top_p=0.95,
)

if __name__ == "__main__":    
    #score, elapsed = bench.run_ifbench(generate, responses_path = ifbench_json_path, workers = 1)
    #print(f"[INFO] | {model_name} | {score:.1f} | {elapsed:.0f} sec")
    #score, elapsed = bench.run_gpqa_diamond(generate, responses_path = diamond_json_path, workers = 1)
    #print(f"[INFO] | {model_name} | {score:.1f} | {elapsed:.0f} sec")
    score, elapsed = bench.run_aime(generate, responses_path = aime_json_path, workers = 1)
    print(f"[INFO] | {model_name} | {score:.1f} | {elapsed:.0f} sec")
    
