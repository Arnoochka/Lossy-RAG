from utils.timer import timer_summary, timer_restart

from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch
from openai import OpenAI

def hf_transformers_model_generate_single_answer(
    model_name: str = "HuggingFaceTB/SmolLM3-3B",
    device: str = "cuda",
    torch_dtype = torch.bfloat16,
    prompt: str = "London is the capital of ",
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 1,
    top_p: float = 0.9,
):
    timer_restart()
    
    device = torch.device(device)
    print(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch_dtype,
        device_map=device,
    ).to(device)

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to(device)

    if do_sample == True:        
        outputs = model.generate(
            **inputs,
            max_length=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
    else:
        outputs = model.generate(
            **inputs,
            max_length=max_new_tokens,
            do_sample=do_sample,
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    timer_summary()
    
    return answer

#before call launch "basalt/bin/basalt serve --config models/basalt_config.json" in cmd
#or
#strata launch #and choose model Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS
#or
#before call launch "llama.cpp/build/bin/llama-server -m models/Qwen3.8-27B-UD-Q3_K_XL.gguf --port 8081 -ngl 99 -c 8192" in cmd

def opeanai_model_generate_single_answer(
    model_name: str = "Qwen3.8-Flash-Next GSQ-RCO Q2_0",
    base_url: str = "http://127.0.0.1:8081/v1",
    prompt: str = "London is the capital of ",
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 1,
    top_p: float = 0.9,
):
    timer_restart()

    client = OpenAI(base_url=base_url, api_key="none")

    # Greedy decoding ≈ temperature 0; the server ignores top_p in that case
    sampling = {"temperature": temperature, "top_p": top_p} if do_sample else {"temperature": 0.0}

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens,
        **sampling,
    )

    answer = response.choices[0].message.content

    timer_summary()
    print(response)
    return answer

if __name__ == "__main__":
    answer = basalt_model_generate_single_answer()
    print(answer)