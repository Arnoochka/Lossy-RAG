from utils.timer import timer_summary, timer_restart

from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch

def model_generate_single_answer(
    model_name: str = "HuggingFaceTB/SmolLM2-1.7B",
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

if __name__ == "__main__":
    answer = model_generate_single_answer()
    print(answer)