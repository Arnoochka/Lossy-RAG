"""Запуск моделей на бенчмарках: HLE, GPQA Diamond, Aime, IFBench. (Squad запускается тут jupyter/stage3_squad_bench.ipynb)"""

from functools import partial

import bench

#NOTE: Cписок моделей
#strata
#Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS
#Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS
#llama.cpp
#Qwen3.8-27B-UD-Q3_K_XL.gguf

#NOTE: Соответствующие функции запуска
# $strata launch --family swift --model IQ2_XS --port 8081 --parallel 8 --context 5k
# $strata launch --family qwen --model IQ2_XS --port 8081 --parallel 8 --context 5k
# $llama.cpp/build/bin/llama-server -m models/Qwen3.8-27B-UD-Q3_K_XL.gguf --port 8081 -np 1 -c 5120 -ngl all -fa on --reasoning-budget 3072

# Пример вычисления параметров
# reasoning_budget_tokens = 3072 (in llama.cpp or include the function opeanai_model_generate_single_answer)
# max_new_tokens = reasoning_budget_tokens + 1024 = 4096
# prompt_tokens = 1024
# max_context_window = max_new_tokens + prompt_tokens = 4096 + 1024 = 5120
# -c for llama.cpp = workers * max_context_window = 1 * 5120 = 5120
# strata context is for each worker only => 5120 = 5k

#NOTE: Тут важно поменять модель
model_name = "Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS"

generate = partial(
    bench.opeanai_model_generate_single_answer,
    model_name=model_name,
    max_new_tokens=4096,
    reasoning_budget_tokens=3072,
    do_sample=True,
    temperature=1,
    top_p=0.95,
)

#NOTE: Логи запуска хранятся в log/model_answers
if __name__ == "__main__":
    runs = (
        (bench.run_ifbench, f"log/model_answers/ifbench_{model_name}.jsonl"),
        (bench.run_gpqa_diamond, f"log/model_answers/gpqa_diamond_{model_name}.jsonl"),
        (bench.run_aime, f"log/model_answers/aime_{model_name}.jsonl"),
    )
    for run, path in runs:
        score, elapsed = run(generate, responses_path=path, workers=1)
        bench.append_time_spend(path, elapsed)
        print(f"[INFO] | {model_name} | {score:.1f} | {elapsed:.0f} sec")
    
    #bench.bench_summary(log_dir="log/model_answers") # Для создания графиков GPQA Diamond, Aime, IFBench.
    #bench.squad_summary(log_dir="log/model_answers") # Для создания графиков squad.
