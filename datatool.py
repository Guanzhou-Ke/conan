import os
import sys
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

DEFAULT_DATA_ROOT = './data'
PROCESSED_DATA_ROOT = os.path.join(DEFAULT_DATA_ROOT, 'processed')
RAW_DATA_ROOT = os.path.join(DEFAULT_DATA_ROOT, 'raw')


def export_dataset(name, views, labels):
    """
    Save dataset as .npz files
    :param name:
    :param views:
    :param labels:
    :return:
    """
    os.makedirs(PROCESSED_DATA_ROOT, exist_ok=True)
    file_path = os.path.join(PROCESSED_DATA_ROOT, f"{name}.npz")
    npz_dict = {"labels": labels, "n_views": len(views)}
    for i, v in enumerate(views):
        npz_dict[f"view_{i}"] = v
    np.savez(file_path, **npz_dict)


def image_edge(img):
    """
    :param img:
    :return:
    """
    img = np.array(img)
    dilation = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=1)
    edge = dilation - img
    return np.stack((img, edge), axis=-1)


def _mnist(dataset_class):
    img_transforms = transforms.Compose([image_edge,
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,))])
    dataset = dataset_class(root=RAW_DATA_ROOT, train=True,
                            download=True, transform=img_transforms)

    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
    data, labels = list(loader)[0]
    return data, labels


def emnist():
    data, labels = _mnist(torchvision.datasets.MNIST)
    views = np.split(data, data.shape[1], axis=1)
    export_dataset("emnist", views=views, labels=labels)


def fmnist():
    data, labels = _mnist(torchvision.datasets.FashionMNIST)
    views = np.split(data, data.shape[1], axis=1)
    export_dataset("fmnist", views=views, labels=labels)


def coil(n_objs=20):
    from skimage.io import imread
    assert n_objs in [20, 100]
    data_dir = os.path.join(RAW_DATA_ROOT, f"coil-{n_objs}")
    img_size = (1, 128, 128) if n_objs == 20 else (3, 128, 128)
    n_imgs = 72
    n_views = 3

    n = (n_objs * n_imgs) // n_views

    views = []
    labels = []

    img_idx = np.arange(n_imgs)

    for obj in range(n_objs):
        obj_list = []
        obj_img_idx = np.random.permutation(img_idx).reshape(n_views, n_imgs // n_views)
        labels += (n_imgs // n_views) * [obj]

        for view, indices in enumerate(obj_img_idx):
            sub_view = []
            for i, idx in enumerate(indices):
                if n_objs == 20:
                    fname = os.path.join(data_dir, f"obj{obj + 1}__{idx}.png")
                    img = imread(fname)[None, ...]
                else:
                    fname = os.path.join(data_dir, f"obj{obj + 1}__{idx * 5}.png")
                    img = imread(fname)
                if n_objs == 100:
                    img = np.transpose(img, (2, 0, 1))
                sub_view.append(img)
            obj_list.append(np.array(sub_view))
        views.append(np.array(obj_list))
    views = np.array(views)
    views = np.transpose(views, (1, 0, 2, 3, 4, 5)).reshape(n_views, n, *img_size)
    labels = np.array(labels)
    export_dataset(f"coil-{n_objs}", views=views, labels=labels)


def _load_npz(name):
    return np.load(os.path.join(PROCESSED_DATA_ROOT, f"{name}.npz"))


class MultiviewDataset(Dataset):

    def __init__(self, views, labels, transform=None):
        self.data = views
        self.targets = torch.LongTensor(labels)
        self.transform = transform
        self.num_view = len(self.data)

    def __getitem__(self, idx):
        views = [self.data[v][idx].float() for v in range(self.num_view)]
        if self.transform is not None:
            views = [self.transform(view) for view in views]
        return views, self.targets[idx]

    def __len__(self):
        return len(self.targets)


def load_dataset(name, img_size=28):
    npz = _load_npz(name)
    labels = npz["labels"]
    views = [npz[f"view_{i}"] for i in range(npz["n_views"])]
    views = [torch.tensor(v) for v in views]

    if name in ['emnist', 'fmnist', 'coil-20', 'coil-100']:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
    else:
        transform = None
    dataset = MultiviewDataset(views, labels, transform)
    return dataset


if __name__ == '__main__':
    dataset = load_dataset('fmnist')
    random_targets = torch.randint(0, 10, (len(dataset), )).numpy()
    print(random_targets.shape)
    gt = dataset.targets.numpy()
    print(gt.shape)
    from util import measure_cluster
    print(measure_cluster(random_targets, gt))
