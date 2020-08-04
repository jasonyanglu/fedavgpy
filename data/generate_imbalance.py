import torch
import numpy as np
import pickle
import os
import torchvision
import argparse
from random import shuffle
cpath = os.path.dirname(__file__)

SAVE = True
np.random.seed(6)


def read_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset',
                        help='name of dataset (mnist, cifar10);',
                        type=str,
                        default='mnist')
    parser.add_argument('--use_1d_feature',
                        action='store_true',
                        default=False,
                        help='represent the image data by 1d feature')
    parser.add_argument('--imbalance_ratio',
                        help='imbalance ratio from 0 to 1;',
                        type=float,
                        default=0.1)
    parser.add_argument('--num_client',
                        help='number of clients',
                        default=100,
                        type=int)

    parsed = parser.parse_args()
    options = parsed.__dict__

    return options


class ImageDataset(object):
    def __init__(self, images, labels, options, normalize=False):
        if isinstance(images, torch.Tensor):
            if options['use_1d_feature']:
                self.data = images.view(-1, 784).numpy()/255
            else:
                self.data = images.numpy()
        else:
            self.data = images
        if normalize and options['use_1d_feature']:
            mu = np.mean(self.data.astype(np.float32), 0)
            sigma = np.std(self.data.astype(np.float32), 0)
            self.data = (self.data.astype(np.float32) - mu) / (sigma + 0.001)
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        self.target = labels

    def __len__(self):
        return len(self.target)


def data_split_imbalance(data, num_split, imbalance_raio, random=True):

    if random:
        np.random.shuffle(data)
    data_num = len(data)
    min_data_num = round(data_num * imbalance_raio)
    min_data = data[:min_data_num]
    maj_data = data[min_data_num:]

    delta, r = len(maj_data) // num_split, len(maj_data) % num_split
    maj_data_lst = []
    i, used_r = 0, 0
    while i < len(maj_data):
        if used_r < r:
            maj_data_lst.append(maj_data[i:i+delta+1])
            i += delta + 1
            used_r += 1
        else:
            maj_data_lst.append(maj_data[i:i+delta])
            i += delta

    delta, r = len(min_data) // num_split, len(min_data) % num_split
    min_data_lst = []
    i, used_r = 0, 0
    while i < len(min_data):
        if used_r < r:
            min_data_lst.append(min_data[i:i+delta+1])
            i += delta + 1
            used_r += 1
        else:
            min_data_lst.append(min_data[i:i+delta])
            i += delta

    return [maj_data_lst, min_data_lst]


def choose_two_digit_imbalance(num_class, num_split):
    split_list = []
    for i in range(num_split):
        while 1:
            x = [i for i in range(num_class)]
            shuffle(x)
            y = [i for i in range(num_class)]
            shuffle(y)
            if sum(np.array(x) == np.array(y)) == 0:
                break
        for xx, yy in zip(x, y):
            split_list.append([xx, yy])

    return split_list


def main():

    options = read_options()
    dataset_file_path = os.path.join(cpath, options['dataset'] + '/data')

    # Get data, normalize, and divide by level
    if options['dataset'] == 'mnist':
        print('>>> Get MNIST data.')
        trainset = torchvision.datasets.MNIST(dataset_file_path, download=True, train=True)
        testset = torchvision.datasets.MNIST(dataset_file_path, download=True, train=False)
    elif options['dataset'] == 'cifar10':
        print('>>> Get CIFAR10 data.')
        trainset = torchvision.datasets.CIFAR10(dataset_file_path, download=True, train=True)
        testset = torchvision.datasets.CIFAR10(dataset_file_path, download=True, train=False)
    else:
        raise Exception('Unknown dataset.')

    train = ImageDataset(trainset.data, trainset.targets, options)
    test = ImageDataset(testset.data, testset.targets, options)

    num_class = len(np.unique(train.target))
    assert(options['num_client'] % num_class == 0)
    num_split = int(options['num_client'] / num_class)
    traindata = []
    for number in range(num_class):
        idx = train.target == number
        traindata.append(train.data[idx])

    split_traindata = []  # num_class x 2 (maj and min) x num_split x num_data x num_feature
    for digit in traindata:
        split_traindata.append(data_split_imbalance(digit, num_split, options['imbalance_ratio']))

    testdata = []
    for number in range(num_class):
        idx = test.target == number
        testdata.append(test.data[idx])
    split_testdata = []
    for digit in testdata:
        split_testdata.append(data_split_imbalance(digit, num_split, options['imbalance_ratio']))

    data_distribution = np.array([len(v) for v in traindata])
    data_distribution = np.round(data_distribution / data_distribution.sum(), 3)
    print('>>> Train Number distribution: {}'.format(data_distribution.tolist()))

    maj_digit_count = np.array([len(v[0]) for v in split_traindata])
    min_digit_count = np.array([len(v[1]) for v in split_traindata])
    print('>>> Each digit in train data is split into majority class {} and minority class {}'.
          format(maj_digit_count.tolist(), min_digit_count.tolist()))

    maj_digit_count = np.array([len(v[0]) for v in split_testdata])
    min_digit_count = np.array([len(v[1]) for v in split_testdata])
    print('>>> Each digit in test data is split into majority class {} and minority class {}'.
          format(maj_digit_count.tolist(), min_digit_count.tolist()))

    # Assign train samples to each user
    train_X = [[] for _ in range(options['num_client'])]
    train_y = [[] for _ in range(options['num_client'])]
    test_X = [[] for _ in range(options['num_client'])]
    test_y = [[] for _ in range(options['num_client'])]

    split_list = choose_two_digit_imbalance(num_class, num_split)

    for user in range(options['num_client']):
        print(user, [[len(v[0]) for v in split_traindata], [len(v[1]) for v in split_traindata]])
        chosen_classes = split_list[user]
        for i, d in enumerate(chosen_classes):
            l = len(split_traindata[d][i][-1])
            train_X[user] += split_traindata[d][i].pop().tolist()
            train_y[user] += (d * np.ones(l)).tolist()

            l = len(split_testdata[d][i][-1])
            test_X[user] += split_testdata[d][i].pop().tolist()
            test_y[user] += (d * np.ones(l)).tolist()

    image = 1 if not options['use_1d_feature'] else 0
    train_path = '{}/data/train/all_data_{}_imbalance.pkl'.format(os.path.join(cpath, options['dataset']), image)
    test_path = '{}/data/test/all_data_{}_imbalance.pkl'.format(os.path.join(cpath, options['dataset']), image)

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Create data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    # Setup 1000 users
    for i in range(options['num_client']):
        uname = i

        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': train_X[i], 'y': train_y[i]}
        train_data['num_samples'].append(len(train_X[i]))

        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': test_X[i], 'y': test_y[i]}
        test_data['num_samples'].append(len(test_X[i]))

    print('>>> User data distribution: {}'.format(train_data['num_samples']))
    print('>>> Total training size: {}'.format(sum(train_data['num_samples'])))
    print('>>> Total testing size: {}'.format(sum(test_data['num_samples'])))

    # Save user data
    if SAVE:
        with open(train_path, 'wb') as outfile:
            pickle.dump(train_data, outfile)
        with open(test_path, 'wb') as outfile:
            pickle.dump(test_data, outfile)

        print('>>> Save data.')


if __name__ == '__main__':
    main()

