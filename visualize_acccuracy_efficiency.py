import matplotlib.pyplot as plt
import numpy as np
import os

MODEL_TYPE="bert"  # bert or roberta
MODEL_SIZE="large"  # base or large
DATASET="RTE"  # SST-2, MRPC, RTE, QNLI, QQP, or MNLI

MODEL_NAME= f"{MODEL_TYPE}-{MODEL_SIZE}"

def flatten(l):
    return [item for sublist in l for item in sublist]

def get_result_idx():
    if DATASET == "MRPC":
        return 1
    else:
        return 0

def pareto_frontier(Xs, Ys, maxX = True, maxY = True):
    # Sort the list in either ascending or descending order of X
    myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
    # Start the Pareto frontier with the first value in the sorted list
    p_front = [myList[0]]    
    # Loop through the sorted list
    for pair in myList[1:]:
        if maxY: 
            if pair[1] >= p_front[-1][1]: # Look for higher values of Y…
                p_front.append(pair) # … and add them to the Pareto frontier
        else:
            if pair[1] <= p_front[-1][1]: # Look for lower values of Y…
                p_front.append(pair) # … and add them to the Pareto frontier
    # Turn resulting pairs back into a list of Xs and Ys
    p_frontX = [pair[0] for pair in p_front]
    p_frontY = [pair[1] for pair in p_front]
    return p_frontX, p_frontY

static = {
    "CoLA": 0.605839001376474,
    "RTE": 0.7148014440433214,
    "MRPC": 0.8937607204116638,
    "SST-2": 0.930045871559633
}

static_accuracy = static[DATASET] 

# Read data
weights = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
my_data = []
for weight in weights:
    results_file = f"saved_models/{MODEL_NAME}/{DATASET}/summary_weight{weight}.csv"
    if not os.path.isfile(results_file):
        continue
    my_data.append(np.genfromtxt(results_file, delimiter=','))
    
ref_data = np.genfromtxt(f'deebert_results_{DATASET}.csv', delimiter=',')

# Plot static model
plt.axhline(y = 100*static_accuracy, linestyle="dotted", color='gray')
plt.scatter((100), (100*static_accuracy), marker="*", color='red', s=80, label='Static Model')

# Plot our results
for i, weight in enumerate(weights):
    plt.scatter(my_data[i][:, -1], 100*my_data[i][:, get_result_idx()], label=f"Ours(w={weight})")

# Plot DeeBERT
plt.plot(ref_data[:, -1], 100*ref_data[:, get_result_idx()], marker="*", label="DeeBert")


# Extract and plot pareto front
Ys = flatten([100*data[:,get_result_idx()] for data in my_data])
Xs = flatten([data[:,-1] for data in my_data])

p_front = pareto_frontier(Xs, Ys, maxX = False, maxY = True) 
plt.plot(p_front[0], p_front[1], linestyle="dashed", color='gray')


plt.xlabel("Computation (%)")
plt.ylabel("Accuracy (%)")
plt.title(f"{DATASET}")
plt.grid()
plt.legend()

plt.savefig(f"vis_{DATASET}_{MODEL_NAME}.png")