
import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer


def hf_transformers_model_generate_single_answer(
    model_name: str = "Qwen/Qwen3.5-4B",
    device: str = "cuda",
    torch_dtype = torch.bfloat16,
    prompt: str = "London is the capital of ",
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: float = 1,
    top_p: float = 0.95,
):  
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
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
    else:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return answer

# $strata launch --family swift --model IQ2_XS --port 8081 --parallel 8 --context 5k
# $strata launch --family qwen --model IQ2_XS --port 8081 --parallel 8 --context 5k

# $llama.cpp/build/bin/llama-server -m models/Qwen3.8-27B-UD-Q3_K_XL.gguf --port 8081 -np 1 -c 5120 -ngl all -fa on --reasoning-budget 3072

# reasoning_budget_tokens = 3072 (in llama.cpp or incide the function opeanai_model_generate_single_answer)
# max_new_tokens = reasoning_budget_tokens + 1024 = 4096
# prompt_tokens = 1024
# max_context_window = max_new_tokens + prompt_tokens = 4096 + 1024 = 5120
# -c for llama.cpp = workers * max_context_window = 1 * 5120 = 5120
# strata context is for each worker only => 5120 = 5k

def opeanai_model_generate_single_answer(
    model_name: str = "Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS",
    base_url: str = "http://127.0.0.1:8081/v1",
    prompt: str = "London is the capital of ",
    max_new_tokens: int = 4096,
    reasoning_budget_tokens: int | None = 3072,
    do_sample: bool = True,
    temperature: float = 1,
    top_p: float = 0.95,
):
    client = OpenAI(base_url=base_url, api_key="none")

    sampling = {"temperature": temperature, "top_p": top_p} if do_sample else {"temperature": 0.0}

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens,
        extra_body= {} if reasoning_budget_tokens is None else {"reasoning_budget_tokens": reasoning_budget_tokens},
        **sampling,
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    answer = opeanai_model_generate_single_answer()
    print(answer)