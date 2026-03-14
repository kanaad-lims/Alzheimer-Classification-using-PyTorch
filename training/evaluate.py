import torch
from sklearn.metrics import classification_report, confusion_matrix

def evaluate(model, dataloader, device):

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.numpy())
            y_pred.extend(predicted.cpu().numpy())

    return y_true, y_pred