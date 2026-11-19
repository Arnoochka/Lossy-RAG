import pandas as pd
import re
import torch
import ast

from tqdm.auto import tqdm
from functools import partial


questions = pd.read_csv("data/bge-large-en-v1.5_Qwen3-Reranker-0.6B_30_3/questions_with_chunks.csv")
questions_paraphrased = pd.read_csv("data/bge-large-en-v1.5_Qwen3-Reranker-0.6B_30_3/questions_paraphrased_with_chunks.csv")

for col in ["answer_text", "answer_start_pos"]:
    questions[col] = questions[col].apply(ast.literal_eval)
    questions_paraphrased[col] = questions_paraphrased[col].apply(ast.literal_eval)

import src.bench as bench
from src.utils.model_generate import opeanai_model_generate_single_answer
from IPython.display import display
from ipywidgets import Output

model_name = "Qwen3.8-27B-UD-Q3_K_XL"
question_num = 50

gen = partial(
    opeanai_model_generate_single_answer,
    model_name=model_name,
    max_new_tokens=4096,
    reasoning_budget_tokens=3072,
    do_sample=True,
    temperature=1,
    top_p=0.95,
    top_k=20,
)

print(f"Model {model_name} score without context", "Num questions", question_num)
score, elapsed = bench.run_squad(gen, questions.head(question_num), responses_path=f"log/model_answers/squad_no_context_{model_name}.jsonl")
score_p, elapsed_p = bench.run_squad(gen, questions_paraphrased.head(question_num), responses_path=f"log/model_answers/squad_p_no_context_{model_name}.jsonl")

print("─" * 80)
print("Score:", score, "\n", "Score paraphrase", score_p)
print("─" * 80)


print(f"Model {model_name} score with full context", "Num questions", question_num)
score, elapsed = bench.run_squad(gen, questions.head(question_num), context_column="context", responses_path=f"log/model_answers/squad_full_context_{model_name}.jsonl")
score_p, elapsed_p = bench.run_squad(gen, questions_paraphrased.head(question_num), context_column="context", responses_path=f"log/model_answers/squad_p_full_context_{model_name}.jsonl")

print("─" * 80)
print("Score:", score, "\n", "Score paraphrase", score_p)
print("─" * 80)


#BAAI/bge-large-en-v1.5, k=30, reranker k=3
print(f"Model {model_name} score with RAG context", "Num questions", question_num)
score, elapsed = bench.run_squad(gen, questions.head(question_num), context_column="chunks", responses_path=f"log/model_answers/squad_rag_{model_name}.jsonl")
score_p, elapsed_p = bench.run_squad(gen, questions_paraphrased.head(question_num), context_column="chunks", responses_path=f"log/model_answers/squad_p_rag_{model_name}.jsonl")

print("─" * 80)
print("Score:", score, "\n", "Score paraphrase", score_p)
print("─" * 80)