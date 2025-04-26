import matplotlib.pyplot as plt
import numpy as np

def plot_losses(train_losses, val_losses):
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(train_losses) + 1), train_losses, label='train Loss', color='cyan', linewidth=2)
    plt.plot(np.arange(1, len(val_losses) + 1), val_losses, label='val loss', color='magenta', linewidth=2)
    plt.xlabel('Epochs', fontsize=12, color='white')
    plt.ylabel('Loss', fontsize=12, color='white')
    plt.legend(fontsize=10)
    plt.grid(False)
    plt.show()
