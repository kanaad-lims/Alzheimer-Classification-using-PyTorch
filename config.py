import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-3

TRAIN_CSV = "data/_train_classes.csv"
VALID_CSV = "data/_valid_classes.csv"
TEST_CSV  = "data/_test_classes.csv"

TRAIN_DIR = "data/train"
VALID_DIR = "data/valid"
TEST_DIR  = "data/test"