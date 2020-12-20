from torch.utils.data import Dataset, DataLoader
import data_transforms
from model import PredictionModel

IMG_HEIGHT = 200
IMG_WIDTH = 200
DEFAULT_DATASET = "dataset/train"
IMAGE_FORMAT = "*.jpg"
LABEL_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G',
               'H', 'I', 'J', 'K', 'L', 'M', 'N',
               'O', 'P', 'Q', 'R', 'S', 'T', 'U',
               'V', 'W', 'X', 'Y', 'Z']


class CustomDataset(Dataset):
    def __init__(self, dataset_path=DEFAULT_DATASET, transform=data_transforms.ToTensor()):
        from PIL import Image
        from glob import glob
        from os import path
        self.data = []

        for f in glob(path.join(dataset_path, IMAGE_FORMAT)):
            i = Image.open(f)
            i.load()
            label = str(f.split('/')[-1])[0]
            label_id = LABEL_NAMES.index(label)
            self.data.append((i, label_id))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = self.transform(*data)
        return data


def load_data(dataset_path=DEFAULT_DATASET, transform=data_transforms.ToTensor(), num_workers=0, batch_size=128):
    custom_dataset = CustomDataset(dataset_path, transform=transform)
    return DataLoader(custom_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, PredictionModel):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'model.th'))

if __name__ == '__main__':
    dataset = load_data()
    for x, y in dataset:
        print(y)
