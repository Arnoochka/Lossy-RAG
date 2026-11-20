# Блок 3
##### Основное описание результатов блока хранится тут _presentation/Stage3.pdf_

##### При запуске используется корневая папка Lossy-RAG
- Для запуска бенчмарков HLE, GPQA Diamond, Aime, IFBench используйте _benchmark/bench_test.py_
- Для запуска бенчмарка squad используйте _jupyter/dataset_intro.ipynb_ потом _data/build_data.py_
_jupyter/stage3_squad_data_preparation.ipynb_ а потом _jupyter/stage3_squad_bench.ipynb_
- Графики из презентации хранятся тут _benchmark/figures/_, получить их можно через вызов __bench.bench_summary()__ и __bench.squad_summary()__

Все логи запусков хранятся в _log/model_answers_ в таком формате:
    _\{Название бенчмарка\}\_{если есть приставка p то вопросы перефразированны}\_\{если бенчмарк squad то написан тип контекста\}\_{название модели}.jsonl_