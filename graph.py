import matplotlib.pyplot as plt
import numpy as np

# Define models and scores
metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
model_1_scores = [0.75, 0.70, 0.78, 0.74]
model_2_scores = [0.90, 0.88, 0.92, 0.90]

x = np.arange(len(metrics))  # label locations
width = 0.3  # bar width

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, model_1_scores, width, label="Model 1", color="red")
rects2 = ax.bar(x + width/2, model_2_scores, width, label="Model 2", color="blue")

# Labels, title, and legend
ax.set_xlabel("Metrics")
ax.set_ylabel("Score")
ax.set_title("Performance Comparison of Models")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

plt.ylim(0, 1)  # Ensure values are between 0-1
plt.show()
