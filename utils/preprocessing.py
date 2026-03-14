import pandas as pd
import torchvision.transforms as transforms

def load_dataframe(csv_path):

    df = pd.read_csv(csv_path)

    class_columns = ["MD","MoD","ND","VMD"]

    df["label"] = df[class_columns].idxmax(axis=1)

    label_mapping = {
                        "MD":0,
                        "MoD":1,
                        "ND":2,
                        "VMD":3
                    }

    df["label_id"] = df["label"].map(label_mapping)
    return df


def get_transforms():
    return transforms.Compose([
        transforms.ToTensor()
    ])