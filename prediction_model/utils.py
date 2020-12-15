from torch.utils.data import Dataset, DataLoader
import data_transforms

IMG_HEIGHT = 200
IMG_WIDTH = 200
DEFAULT_DATASET = "dataset/train"
LABEL_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G',
               'H', 'I', 'J', 'K', 'L', 'M', 'N'
               'O', 'P', 'Q', 'R', 'S', 'T', 'U',
               'V', 'W', 'X', 'Y', 'Z', 'space', 'nothing']


class CustomDataset(Dataset):
    def __init__(self, dataset_path=DEFAULT_DATASET, transform=data_transforms.ToTensor()):
        from PIL import Image
        from glob import glob
        from os import path
        self.data = []

        for f in glob(path.join(dataset_path, '*/*.jpg')):
            i = Image.open(f)
            i.load()
            label = str(f)[0]
            if label == "s":
                label = "space"
            elif label == "n":
                label = "nothing"
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


if __name__ == '__main__':
    dataset = CustomDataset()
    train_data = load_data()
    for x, y in train_data:
        print(x, y)
