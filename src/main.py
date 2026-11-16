import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from .database import Database
from .encoder import BaseEncoder
from .quant.sq import SqQuant
from .search.ivf import IvfSearch


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    texts = [
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.",
        "Rome is the capital of Italy.",
        "Madrid is the capital of Spain.",
        "The Louvre is an art museum in Paris.",
        "The Seine is a river that flows through Paris.",
    ]
    question = "What is the capital of France?"

    encoder_name = "BAAI/bge-small-en-v1.5"
    encoder = BaseEncoder(
        model=AutoModel.from_pretrained(encoder_name).to(device),
        tokenizer=AutoTokenizer.from_pretrained(encoder_name),
        query_instruction="Represent this sentence for searching relevant passages: ",
    )
    searcher = IvfSearch(
        encoder=encoder,
        quantizer=SqQuant(num_bits=8),
        nlist=2,
        nprobe=1,
    )
    database = Database(searcher)
    database.fit(texts[:4])
    database.set_data(texts)

    llm_name = "Qwen/Qwen2.5-0.5B-Instruct"
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)
    llm = AutoModelForCausalLM.from_pretrained(llm_name).to(device).eval()

    def compute_num_tokens(text: str) -> int:
        return len(llm_tokenizer.encode(text, add_special_tokens=False))

    database.set_token_computer(compute_num_tokens)
    results = database.search(question, k=3, max_context_tokens=512)
    context = "\n\n".join(results["text"])

    messages = [
        {
            "role": "system",
            "content": (
                "Answer the question using the provided context. "
                "If the information is insufficient, say so."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}",
        },
    ]
    inputs = llm_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(llm.device)

    with torch.inference_mode():
        outputs = llm.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    answer_tokens = outputs[0, inputs["input_ids"].shape[1]:]
    answer = llm_tokenizer.decode(answer_tokens, skip_special_tokens=True)
    print(f"Context:\n{context}\n\nAnswer:\n{answer}")


if __name__ == "__main__":
    main()
