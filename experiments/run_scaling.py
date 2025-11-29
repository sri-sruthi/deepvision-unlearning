import argparse
import numpy as np
from experiments.run_unlearning import main as run_unlearn

def run_scaling_experiment(model, dataset, baseline_path, method="ga"):
    forget_sizes = [100, 500, 1000, 2000, 5000, 10000]
    results = []

    for size in forget_sizes:
        print(f"Running scaling experiment: forget-count={size}")
        result = run_unlearn(
            model=model,
            dataset=dataset,
            method=method,
            forget_type="sample",
            forget_count=size,
            model_path=baseline_path
        )
        results.append((size, result))

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--baseline-path", type=str, required=True)
    args = parser.parse_args()

    run_scaling_experiment(args.model, args.dataset, args.baseline_path)
