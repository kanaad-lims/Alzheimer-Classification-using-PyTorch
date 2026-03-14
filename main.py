import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.alzheimer_dataset import AlzheimerDataset
from models.alzheimer_model import AlzheimerModel
from training.train import train_epoch
from training.evaluate import evaluate
from utils.preprocessing import load_dataframe, get_transforms
import config

train_df = load_dataframe(config.TRAIN_CSV)
valid_df = load_dataframe(config.VALID_CSV)
test_df = load_dataframe(config.TEST_CSV)

transform = get_transforms()

train_dataset = AlzheimerDataset(train_df, config.TRAIN_DIR, transform)
valid_dataset = AlzheimerDataset(valid_df, config.VALID_DIR, transform)
test_dataset = AlzheimerDataset(test_df, config.TEST_DIR, transform)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)

model = AlzheimerModel().to(config.DEVICE)

loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=config.LR)

for epoch in range(config.EPOCHS):

    train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_function, config.DEVICE)

    print(epoch, train_loss, train_acc)

y_true, y_pred = evaluate(model, test_loader, config.DEVICE)