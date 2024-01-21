import csv
import os
import matplotlib.pyplot as plt
import numpy as np

CSV_DIR = os.path.join("..", "csv_records", "pruning")
OUT_DIR = os.path.join(CSV_DIR, "export")

# Output
EXP_NAME = "Blend"
OUT_FILENAME = "pruning_results_resnet_blend"

# CSV file
CSV_FILENAME = "pruning_resnet_blend_0120_1833"
# CSV_FILENAME = "pruning_resnet_badnets_grid_0120_1709"
# CSV_FILENAME = "pruning_resnet_badnets_square_0120_1751"

layers = [
    "layer4.0.conv1",
    "layer4.0.conv2",
    "layer4.1.conv1",
    "layer4.1.conv2",
]

x = np.linspace(0,1,11)
y_acc_clean = {
    "layer4.0.conv1": None,
    "layer4.0.conv2": None,
    "layer4.1.conv1": None,
    "layer4.1.conv2": None
}
y_asr_targeted = {
    "layer4.0.conv1": None,
    "layer4.0.conv2": None,
    "layer4.1.conv1": None,
    "layer4.1.conv2": None
}
y_asr_untargeted = {
    "layer4.0.conv1": None,
    "layer4.0.conv2": None,
    "layer4.1.conv1": None,
    "layer4.1.conv2": None
}
y_acc_backdoor = {
    "layer4.0.conv1": None,
    "layer4.0.conv2": None,
    "layer4.1.conv1": None,
    "layer4.1.conv2": None
}

with open(os.path.join(CSV_DIR, f"{CSV_FILENAME}.csv")) as outputs:
    reader = csv.reader(outputs)
    headers = next(reader)
    for _ in range(len(layers)):
        curr_acc_clean = []
        curr_asr_untargeted = []
        curr_asr_targeted = []
        for _ in range(11):
            data = next(reader)
            curr_acc_clean.append(float(data[3]))
            curr_asr_targeted.append(float(data[6]))
            curr_asr_untargeted.append(float(data[5]))

        y_acc_clean[data[2]] = np.array(curr_acc_clean)
        y_asr_targeted[data[2]] = np.array(curr_asr_targeted)
        y_asr_untargeted[data[2]] = np.array(curr_asr_untargeted)*100


fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, figsize=(12,12), layout="constrained")
ax = [ax1,ax2,ax3,ax4]
for i, layer in enumerate(layers):
    ax[i].plot(x,y_acc_clean[layer], label="Accuracy (clean)")
    ax[i].plot(x,y_asr_untargeted[layer], label="Untargeted ASR")
    ax[i].plot(x,y_asr_targeted[layer], label="Targeted ASR")
    ax[i].axhline(y=max(y_acc_clean[layer]) - 4, linewidth=1, linestyle=":", color="black")

    ax[i].set_yticks(np.arange(0,101,10))
    ax[i].set_xlabel("Fraction of neurons pruned")
    ax[i].set_ylabel("Rate / %")
    ax[i].set_title(layer)
    ax[i].legend(loc="lower left")

fig.suptitle(f"Pruning - {EXP_NAME} (resnet18)", fontsize=15)

# Save figure
os.makedirs(OUT_DIR, exist_ok=True)
plt.savefig(os.path.join(OUT_DIR,f"{OUT_FILENAME}.svg"))
plt.savefig(os.path.join(OUT_DIR,f"{OUT_FILENAME}.png"))
plt.show()