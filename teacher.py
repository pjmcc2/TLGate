from gate import train
from gate import test
from torch import nn
import torch
import numpy as np
from torchvision.models import vgg13
from tqdm import tqdm
import pickle
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
import torch.optim as optim
import argparse


def main(args):
    epochs = int(args.epochs)
    num_tests = int(args.num_tests)
    batch_size = int(args.batch_size)

    train_data = CIFAR10(root="data",
                         train=True,
                         download=True,
                         transform=Compose([ToTensor(), Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]))

    test_data = CIFAR10(root="data",
                        train=False,
                        download=True,
                        transform=Compose([ToTensor(), Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]))

    batch_size = batch_size

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'




    ### training part ###

    criterion = nn.CrossEntropyLoss()

    acc_lists_train = []
    loss_lists_train = []
    acc_lists_test = []
    loss_lists_test = []
    instance_check = None
    for i in range(num_tests):
        teacher = vgg13(weights='DEFAULT', progress=True)
        if instance_check is not None and (not np.allclose(teacher.classifier[0].weight.detach().numpy(), instance_check)):
            raise ValueError("reinstantiation failed")
        else:
            instance_check = teacher.classifier[0].weight.detach().numpy()
        teacher.classifier[6] = nn.Linear(4096, 10, bias=True)
        teacher.to(device)
        optimizer = optim.Adam(teacher.parameters(), lr=0.0001)
        a_list_train = []
        a_list_test = []
        loss_list_train = []
        loss_list_test = []

        for epoch in range(epochs): # test teacher separately
            acc, loss = train(teacher,train_dataloader, criterion, optimizer,device)
            a_list_train.append(acc)
            loss_list_train.append(loss)

            acc, loss = test(teacher, test_dataloader, criterion, device)

            a_list_test.append(acc)
            loss_list_test.append(loss)

        acc_lists_test.append(a_list_test)
        acc_lists_train.append(a_list_train)
        loss_lists_test.append(loss_list_test)
        loss_lists_train.append(loss_list_train)

    tr_acc = np.array(acc_lists_train)
    te_acc = np.array(acc_lists_test)
    tr_loss = np.array(loss_lists_train)
    te_loss = np.array(loss_lists_test)

    save_dict = {"train_acc": tr_acc,
                 "test_acc": te_acc,
                 "train_loss": tr_loss,
                 "test_loss": te_loss,
                 "model": teacher
                 }
    with open(f"pickles/teacher.pickle", "wb") as f:
        pickle.dump(save_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a KD model and pickle the results")
    parser.add_argument("epochs", help="Number of training epochs")
    parser.add_argument("num_tests", help="number of training iterations")
    parser.add_argument("batch_size", help="batch_size")

    args = parser.parse_args()
    main(args)
