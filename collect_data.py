import numpy as np
import csv

MODEL_TYPE="bert"  # bert or roberta
MODEL_SIZE="base"  # base or large
DATASET="CoLA"  # SST-2, MRPC, RTE, QNLI, QQP, or MNLI

MODEL_NAME= f"{MODEL_TYPE}-{MODEL_SIZE}"


for weight in [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1, 1.5]:
    results = []

    for target in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        file_name = f"saved_models/{MODEL_NAME}/{DATASET}/ada_weight{weight}/ada_target{target}/eval_results.txt"
        with open(file_name) as f:
            lines = f.readlines()
            result = []
            for line in lines:
                line = line.split()
                result.append(float(line[-1]))
            results.append(result)


    with open(f"saved_models/{MODEL_NAME}/{DATASET}/summary_weight{weight}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(results)

        