import numpy as np
import torch
import os
import torchvision
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from src.models.model import CifarCnn
from src.optimizers.gd import GD

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

cpath = os.path.dirname(__file__)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
IMAGE_DATA = True
IMAGE_SIZE = [32, 32]
DATASET_FILE = os.path.join(cpath, 'data/cifar10/data')

class ImageDataset(object):
    def __init__(self, images, labels, normalize=False):
        if isinstance(images, torch.Tensor):
            if not IMAGE_DATA:
                self.data = images.view(-1, IMAGE_SIZE[0] * IMAGE_SIZE[1]).numpy()/255
            else:
                self.data = images.numpy()
        else:
            self.data = images
        if normalize and not IMAGE_DATA:
            mu = np.mean(self.data.astype(np.float32), 0)
            sigma = np.std(self.data.astype(np.float32), 0)
            self.data = (self.data.astype(np.float32) - mu) / (sigma + 0.001)
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        self.target = labels

    def __len__(self):
        return len(self.target)


class MiniDataset(Dataset):
    def __init__(self, data, labels):
        super(MiniDataset, self).__init__()
        self.data = np.array(data)
        self.labels = np.array(labels).astype("int64")

        if self.data.ndim == 4 and self.data.shape[3] == 3:
            self.data = self.data.astype("uint8")
            self.transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                 transforms.RandomCrop(32, 4),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                 ]
            )
        elif self.data.ndim == 4 and self.data.shape[3] == 1:
            self.transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))
                 ]
            )
        elif self.data.ndim == 3:
            self.data = self.data.reshape(-1, 28, 28, 1).astype("uint8")
            self.transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))
                 ]
            )
        else:
            self.data = self.data.astype("float32")
            self.transform = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data, target = self.data[index], self.labels[index]

        if self.data.ndim == 4 and self.data.shape[3] == 3:
            data = Image.fromarray(data)

        if self.transform is not None:
            data = self.transform(data)

        return data, target


class BatchCNN:
    def __init__(self, train_data, test_data):
        self.model = CifarCnn((3, 32, 32), 10)
        self.optimizer = GD(self.model.parameters(), lr=0.1, weight_decay=0.001)
        self.batch_size = 64
        self.num_epoch = 100
        self.train_dataloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
        self.criterion = CrossEntropyLoss()

    def train(self):

        self.model.train()
        train_loss = train_acc = train_total = 0
        for epoch in range(self.num_epoch):
            for batch_idx, (x, y) in enumerate(self.train_dataloader):
                # print('%d ' % batch_idx, end='')
                self.optimizer.zero_grad()
                pred = self.model(x)

                loss = self.criterion(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)
                self.optimizer.step()

                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()
                target_size = y.size(0)

                train_loss += loss.item() * y.size(0)
                train_acc += correct
                train_total += target_size

            print("Epoch: {:>2d} | train loss {:>.4f} | train acc {:>5.2f}%".format(
                   epoch, train_loss/train_total, train_acc/train_total*100))

            if epoch % 5 == 0:
                test_loss = test_acc = test_total = 0.
                with torch.no_grad():
                    for x, y in self.test_dataloader:

                        pred = self.model(x)
                        loss = self.criterion(pred, y)
                        _, predicted = torch.max(pred, 1)
                        correct = predicted.eq(y).sum()

                        test_acc += correct.item()
                        test_loss += loss.item() * y.size(0)
                        test_total += y.size(0)

                print("Epoch: {:>2d} | test loss {:>.4f} | test acc {:>5.2f}%".format(
                    epoch, test_loss / test_total, test_acc / test_total * 100))


def main():

    trainset = torchvision.datasets.CIFAR10(DATASET_FILE, download=True, train=True)
    testset = torchvision.datasets.CIFAR10(DATASET_FILE, download=True, train=False)

    train = ImageDataset(trainset.data, trainset.targets)
    test = ImageDataset(testset.data, testset.targets)

    train_data = MiniDataset(train.data, train.target)
    test_data = MiniDataset(train.data, train.target)

    # Call appropriate trainer
    trainer = BatchCNN(train_data, test_data)
    trainer.train()


if __name__ == '__main__':
    main()