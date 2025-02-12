import matplotlib.pyplot as plt
import numpy as np

# Sample data: Test case (question) numbers
test_cases = np.arange(1, 11)

# Accuracy data for Model 1 and Model 2 (simulated)
accuracy_model_1 = [0.75, 0.80, 0.70, 0.65, 0.76, 0.72, 0.80, 0.78, 0.74, 0.80]  # Model 1 accuracy
accuracy_model_2 = [0.90, 0.88, 0.92, 0.91, 0.89, 0.85, 0.93, 0.91, 0.90, 0.92]  # Model 2 accuracy

# Response time data for Model 1 and Model 2 (in milliseconds)
response_time_model_1 = [500, 450, 520, 510, 490, 480, 500, 530, 510, 495]  # Model 1 response time (ms)
response_time_model_2 = [200, 180, 190, 170, 160, 175, 180, 190, 185, 170]  # Model 2 response time (ms)

# Create figure and axis
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot accuracy on the first y-axis
ax1.set_xlabel('Test Cases')
ax1.set_ylabel('Accuracy', color='tab:blue')
ax1.plot(test_cases, accuracy_model_1, label='Llama3 Accuracy', color='tab:blue', marker='o')
ax1.plot(test_cases, accuracy_model_2, label='Qwen2 Accuracy', color='tab:green', marker='o')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create a second y-axis for response time
ax2 = ax1.twinx()
ax2.set_ylabel('Response Time (ms)', color='tab:red')
ax2.plot(test_cases, response_time_model_1, label='Llama3 Response Time', color='tab:red', linestyle='--')
ax2.plot(test_cases, response_time_model_2, label='Qwen2 Response Time', color='tab:orange', linestyle='--')
ax2.tick_params(axis='y', labelcolor='tab:red')

# Title and legends
plt.title('Performance Comparison of Llama3 and Qwen2')
fig.tight_layout()
fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
plt.grid(True)
plt.show()
