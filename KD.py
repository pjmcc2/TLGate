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




class studentNet(nn.Module):
    def __init__(self, teacher, T):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1),  # 30,30,64
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, 1),  # 28,128
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, 2, padding=1),  # 14,128
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 14, 1),
            nn.Flatten(),
            nn.Linear(128, 1000),
            nn.ReLU()
        )

        self.linear3 = nn.Linear(1000, 1000)
        self.linear2 = nn.Linear(1000, 1000)
        self.linear1 = nn.Linear(1000, 10)

        self.teacher = teacher
        self.T = T

    def forward(self, X):
        student_logits = self.feature_extractor(X)
        teacher_logits = self.teacher(X)
        teacher_prob = nn.functional.softmax(teacher_logits / self.T, dim=-1)
        student_prob = nn.functional.softmax(student_logits / self.T, dim=-1)
        return student_prob, teacher_prob


## DISCLAIMER: THE FOLLOWING CODE IS MOSTLY COPIED FROM THE PYTORCH KD TUTORIAL ##
def train_knowledge_distillation(teacher, student, train_loader, optim, loss_fn, T, a, device):
    teacher.eval()  # Teacher set to evaluation mode
    student.train()  # Student to train mode
    optimizer = optim
    inc_acc=0
    inc_loss = 0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
        with torch.no_grad():
            teacher_logits = teacher(inputs)

        # Forward pass with the student model
        student_logits = student(inputs)

        # I am changing this forget the log.
        ##Soften the student logits by applying softmax first and log() second
        soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
        # soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)
        soft_prob = nn.functional.softmax(student_logits / T, dim=-1)

        # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
        soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T ** 2)

        # Calculate the true label loss
        label_loss = loss_fn(student_logits, labels)

        # Weighted sum of the two losses
        loss = a * soft_targets_loss + (1 - a) * label_loss
        inc_loss = inc_loss + (loss.item() - inc_loss) / (i + 1)

        loss.backward()
        optimizer.step()
        _, predicted = torch.max(student_logits.data, 1)
        acc = (predicted == labels).sum().item()/inputs.shape[0]
        inc_acc = inc_acc + (acc-inc_acc)/(i+1)
    return inc_acc, inc_loss


def main(args):
    T = args.Temperature
    a = args.a
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

    with open("pickles/teacher.pickle","rb") as f:
        teacher_saved = pickle.load(f)

    teacher = teacher_saved["model"]
    teacher.to(device)
    # Freeze layers
    for param in teacher.parameters():
        param.requires_grad = False

    ### training part ###
    epochs = epochs
    num_tests = num_tests
    criterion = nn.CrossEntropyLoss()

    acc_lists_train = []
    loss_lists_train = []
    acc_lists_test = []
    loss_lists_test = []

    for i in range(num_tests):
        model = studentNet(teacher, -1)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        a_list_train = []
        a_list_test = []
        loss_list_train = []
        loss_list_test = []

        for epoch in range(epochs): # test teacher separately
            acc, loss = train_knowledge_distillation(teacher, model, train_dataloader, optimizer, criterion, T, a, device)
            a_list_train.append(acc)
            loss_list_train.append(loss)

            acc, loss = test(model, test_dataloader, criterion, device)

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
                 "test_loss": te_loss
                 }
    with open(f"pickles/KD_model.pickle", "wb") as f:
        pickle.dump(save_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a KD model and pickle the results")
    parser.add_argument("Temperature", help="Temperature: Softens the distribution")
    parser.add_argument("a", help="weighted average parameter (for soft targets)")
    parser.add_argument("epochs", help="Number of training epochs")
    parser.add_argument("num_tests", help="number of training iterations")
    parser.add_argument("batch_size", help="batch_size")

    args = parser.parse_args()
    main(args)
