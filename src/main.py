import torch
from utils.model_generate import model_generate_single_answer

#meta-llama/Llama-3.2-3B-Instruct ~ 6.5 Gb
#google/gemma-3-4b-pt ~ 8.6 Gb
#Qwen/Qwen3.5-4B ~ 8.4Gb
#HuggingFaceTB/SmolLM3-3B ~ 6.2Gb

if __name__ == "__main__":
    print(
        model_generate_single_answer(
        model_name = "Qwen/Qwen3.5-4B",
        device="cuda",
        torch_dtype = torch.bfloat16,
        prompt = "Cavanaugh's legacy at Notre Dame in the post-war years was devoted to raising academic standards and reshaping the university administration to suit it to an enlarged educational mission and an expanded student body and stressing advanced studies and research at a time when Notre Dame quadrupled in student census, undergraduate enrollment increased by more than half, and graduate student enrollment grew fivefold. Cavanaugh also established the Lobund Institute for Animal Studies and Notre Dame's Medieval Institute. London is the capital of ",
        max_new_tokens = 256,
        do_sample = True,
        )
    )