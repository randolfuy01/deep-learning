import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('./model/cnn_model_test_3.csv') 

epochs = df['Epoch']  
training_losses = df['Training Loss']
validation_losses = df['Validation Loss']
accuracies = df['Accuracy (%)']
learning_rates = df['Learning Rate']

fig, ax1 = plt.subplots(figsize=(12, 6))


ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='tab:blue')
ax1.plot(epochs, training_losses, color='tab:blue', label='Training Loss')
ax1.plot(epochs, validation_losses, color='tab:orange', label='Validation Loss')
ax1.tick_params(axis='y', labelcolor='tab:blue')


ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy (%)', color='tab:green')
ax2.plot(epochs, accuracies, color='tab:green', label='Accuracy')
ax2.tick_params(axis='y', labelcolor='tab:green')


for i, lr in enumerate(learning_rates):
    if i % 20 == 0: 
        ax1.annotate(f'lr:{lr:.6f}', (epochs[i], training_losses[i]),
                     textcoords="offset points", xytext=(0, 25), ha='center', fontsize=8, color='tab:red')


high = 73 
for i, ac in enumerate(accuracies):
    if ac > high: 
        high = ac
        ax2.annotate(f'{ac:.2f}', (epochs[i], accuracies[i]), 
                     textcoords="offset points", xytext=(0, 10), 
                     ha='center', fontsize=8, color='tab:green')


for i, ls in enumerate(validation_losses):
    if i % 20 == 0:
        ax1.annotate(f'{ls:.5f}', (epochs[i], validation_losses[i]), 
                     textcoords="offset points", xytext=(0, -20), 
                     ha='center', fontsize=8, color='tab:blue')


ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.title('Training/Validation Loss, Accuracy, and Learning Rate over Epochs')

fig.tight_layout()


plt.show()
