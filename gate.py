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



# Classifier network
class ClassNet(nn.Module):
  def __init__(self,gate_net,loc=0):
    super().__init__()
    self.feature_extractor = nn.Sequential(
        nn.Conv2d(3,64,3,1), # 30,30,64
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Conv2d(64,128,3,1), # 28,128
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.Conv2d(128,128,3,2,padding=1), # 14,128
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.Conv2d(128,128,14,1),
        nn.Flatten(),
        nn.Linear(128,1000),
        nn.ReLU(),

        )

    self.linear3 = nn.Linear(1000,1000)
    self.linear2 = nn.Linear(1000,1000)
    self.linear1 = nn.Linear(1000,10)
    self.last_layers = [self.linear3, self.linear2,self.linear1]
    self.relu = nn.ReLU()
    self.gate_net = gate_net
    self.gate_location = loc


  def forward(self,input):
    X = self.feature_extractor(input)
    gate = self.gate_net(input)

    if self.gate_location == "all":
      flag = True
    else:
      flag = False
    for i in range(2):
      if flag:
        self.gate_location = i
      if self.training:
        if i == self.gate_location: # check if this location is for the gate
          X = nn.functional.relu(torch.mul(X,gate))
        X = self.relu(self.last_layers[i](X))

      else:
        X = self.relu(self.last_layers[i](X))
    return X


def train(model,dataloader, criterion, optimizer,device='cpu'):
    model.train()
    inc_acc=0
    inc_loss = 0
    for i, data in tqdm(enumerate(dataloader)):
        # get the inputs; data is a list of [inputs, labels]
        X, y = data

        X=X.to(device)
        y=y.to(device)
        optimizer.zero_grad()


        outputs = model(X)
        loss = criterion(outputs, y)
        inc_loss = inc_loss + (loss.item()-inc_loss)/(i+1)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        acc = (predicted == y).sum()/X.shape[0]
        inc_acc = inc_acc + (acc-inc_acc)/(i+1)
    return inc_acc,inc_loss

def test(model,dataloader, criterion,device='cpu'):
    model.eval()
    inc_acc=0
    inc_loss = 0
    for i, data in tqdm(enumerate(dataloader)):
        # get the inputs; data is a list of [inputs, labels]
        X, y = data
        X=X.to(device)
        y=y.to(device)
        
        outputs = model(X)
        loss = criterion(outputs, y)
        inc_loss = inc_loss + (loss.item()-inc_loss)/(i+1)

        _, predicted = torch.max(outputs.data, 1)
        acc = (predicted == y).sum()/X.shape[0]
        inc_acc = inc_acc + (acc-inc_acc)/(i+1)
    return inc_acc,inc_loss


def main(epochs,num_tests,batch_size):
    
    train_data = CIFAR10(root="data",
                        train=True,
                        download=True,
                        transform=Compose([ToTensor(), Normalize(mean=(0.5,0.5,0.5),std = (0.5,0.5,0.5))]))

    test_data = CIFAR10(root="data",
                        train=False,
                        download=True,
                        transform=Compose([ToTensor(), Normalize(mean=(0.5,0.5,0.5),std = (0.5,0.5,0.5))]))


    batch_size = batch_size

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device='cpu'

    gate = vgg13(weights='DEFAULT',progress=True)
    gate.to(device)
    # Freeze layers
    for param in gate.parameters():
        param.requires_grad = False



    gate_model_labels = ["no_gates", "layer_1","layer_2","last_layer", "all"]
    model_params = [-1, 0, 1, 2, "all"]


    ### training part ###
    epochs = epochs
    num_tests = num_tests
    criterion = nn.CrossEntropyLoss()


    for j in range(len(model_params)):
        acc_lists_train = []
        loss_lists_train = []
        acc_lists_test = []
        loss_lists_test = []
        for i in range(num_tests):
            model = ClassNet(gate,model_params[j]) 
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.0001)
            a_list_train = []
            a_list_test = []
            loss_list_train = []
            loss_list_test = []

            for epoch in range(epochs):
                
                loss, acc = train(model,train_dataloader,criterion,optimizer,device)
                a_list_train.append(acc)
                loss_list_train.append(loss)

                loss,acc = test(model,test_dataloader,criterion,optimizer,device)
                
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

        save_dict = {"train_acc":tr_acc,
                    "test_acc":te_acc,
                    "train_loss":tr_loss,
                    "test_loss":te_loss
                    }
        with open(f"pickles/gate_model_{gate_model_labels[j]}.pickle","wb") as f:
            pickle.dump(save_dict,f)

if __name__ == "__main__":
   
   epochs = 1
   num_tests=1
   batch_size=512
   main(epochs,num_tests, batch_size)