import time
import glob
import os
import pandas as pd
import shutil
import torch
from functools import wraps

def timer(layer, additional_path: str, base_dir: str = "model/time_spend") -> Callable:
    def decorator(func):
        file_handles = {}
        layer_getter = layer if callable(layer) else lambda *args, **kwargs: layer

        def get_handle(actual_layer):
            if actual_layer not in file_handles:
                path = f"{base_dir}/{additional_path}/time_spend_{actual_layer}.txt"
                os.makedirs(os.path.dirname(path), exist_ok=True)
                file_handles[actual_layer] = open(path, "a", buffering=1)
            return file_handles[actual_layer]

        @wraps(func)
        def wrapper(*args, **kwargs):
            actual_layer = layer_getter(*args, **kwargs)
            file_handle = get_handle(actual_layer)

            if torch.cuda.is_available():
                # CUDA
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                result = func(*args, **kwargs)
                end_event.record()

                torch.cuda.synchronize()

                duration_sec = start_event.elapsed_time(end_event) / 1e3
            else:
                # CPU
                start = time.perf_counter_ns()
                result = func(*args, **kwargs)
                end = time.perf_counter_ns()
                duration_sec = (end - start) / 1e9

            file_handle.write(f"{duration_sec}\n")
            return result

        return wrapper
    return decorator

def timer_restart(base_dir:str = "model/time_spend") -> None:
    try:
        shutil.rmtree(f"{base_dir}")
    except:
        pass
    return

def timer_summary(base_dir: str = "model/time_spend") -> None:
    os.makedirs(base_dir, exist_ok=True)

    all_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(f"{base_dir}/{d}")]

    for d in all_dirs:
        target_dir = f"{base_dir}/{d}"
        files = glob.glob(f"{target_dir}/time_spend_*.txt")
        total = 0.0
        for path in files:
            try:
                with open(path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            total += float(line)
            except:
                pass
        with open(f"{target_dir}/total.txt", "w") as f:
            f.write(f"{total:.3f}\n")

    data = []
    for d in all_dirs:
        total_file = f"{base_dir}/{d}/total.txt"
        if os.path.exists(total_file):
            try:
                with open(total_file, "r") as f:
                    value = float(f.read().strip())
                    data.append([d, value])
            except:
                pass

    df = pd.DataFrame(data)
    df.to_csv(f"{base_dir}/total.txt", index=False, header=False)

if __name__ == "__main__":
    timer_summary()
    print("Done")