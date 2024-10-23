import matplotlib.pyplot as plt
import re
import math

# Read the log file (Replace 'your_log_file.txt' with your actual file path)
with open("no_tokenizer.log", "r") as file:
    log_data = file.read()

# Step 1: Extract train sections and accuracies
train_sections = re.findall(r"__train_start__\n(.*?)\n__train_stop__", log_data, re.DOTALL)
accuracies = re.findall(r"__evaluate_start__\n(.*?)\n__evaluate_stop__", log_data, re.DOTALL)

# Step 2: Parse each section into epoch-loss data
fold_data = []
for section in train_sections:
    lines = section.strip().split('\n')[1:]  # Skip header (epoch,loss)
    epochs, losses = zip(*(map(float, line.split(',')) for line in lines))
    fold_data.append((epochs, losses))

# Step 3: Create a grid for subplots (e.g., 2x3 for 5 folds)
num_folds = len(fold_data)
cols = 3  # Adjust columns as needed
rows = math.ceil(num_folds / cols)

fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
axes = axes.flatten()  # Flatten for easy iteration

# Step 4: Plot each fold in its own subplot
for i, (epochs, losses) in enumerate(fold_data):
    accuracy = float(accuracies[i])
    ax = axes[i]
    ax.plot(epochs, losses, label=f"Fold {i + 1}")
    ax.set_title(f"Fold {i + 1} (Acc: {accuracy:.2f})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)

# Step 5: Remove any empty subplots (if folds < grid slots)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Step 6: Adjust layout and display the plot
plt.tight_layout()
plt.show()

