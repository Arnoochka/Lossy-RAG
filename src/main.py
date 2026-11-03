import torch
from utils.model_generate import hf_transformers_model_generate_single_answer, opeanai_model_generate_single_answer

#hf_transformers
#meta-llama/Llama-3.2-3B-Instruct ~ 6.5 Gb
#google/gemma-3-4b-pt ~ 8.6 Gb
#Qwen/Qwen3.5-4B ~ 8.4Gb
#HuggingFaceTB/SmolLM3-3B ~ 6.2Gb

#basalt
#Qwen3.8-Flash-Next GSQ-RCO Q2_0

#strata
#Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS

#llama.cpp
#Qwen3.8-27B-UD-Q3_K_XL.gguf

if __name__ == "__main__":    
    #print(hf_transformers_model_generate_single_answer())
    print(opeanai_model_generate_single_answer(prompt="London is the capital of "))