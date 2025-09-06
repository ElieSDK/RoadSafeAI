import torch
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

def save_checkpoint(features, mat_head, qual_head, path):
    torch.save({
        "features": features.state_dict(),
        "mat_head": mat_head.state_dict(),
        "qual_head": qual_head.state_dict()
    }, path)

def plot_confusion_matrices(cm_mat, cm_qual, material_names, quality_names):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm_mat.cpu().numpy(), annot=True, fmt='d', cmap='Blues',
                xticklabels=material_names, yticklabels=material_names, ax=axes[0])
    axes[0].set_title('Confusion Matrix: Material')

    sns.heatmap(cm_qual.cpu().numpy(), annot=True, fmt='d', cmap='Blues',
                xticklabels=quality_names, yticklabels=quality_names, ax=axes[1])
    axes[1].set_title('Confusion Matrix: Quality')
    plt.show()

def plot_epoch_accuracy(epoch_metrics):
    epochs = range(1, len(epoch_metrics) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, epoch_metrics, marker='o')
    plt.title('Average Validation Accuracy per Epoch')
    plt.xlabel('Epoch'); plt.ylabel('Average Accuracy'); plt.grid(True)
    plt.xticks(epochs)
    plt.show()

def print_file_creation(path):
    ctime = datetime.datetime.fromtimestamp(os.path.getctime(path))
    print(f"The file '{path}' was created on: {ctime}")
