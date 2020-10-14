# source code inspireed by
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#model-training-and-validation-code

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root = './data'
transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()
])

train_set = datasets.FashionMNIST(root=root, train=True, transform=transform, download=True)
test_set = datasets.FashionMNIST(root=root, train=False, transform=transform, download=True)

# hyperparameters
batch_size = 16
num_epochs = 15
learning_rate = 0.001
momentum = 0.9

# Load train and test data
data_loaders = {}
data_loaders['train'] = torch.utils.data.DataLoader(
                            dataset=train_set,
                            batch_size=batch_size,
                            shuffle=True)
data_loaders['test'] = torch.utils.data.DataLoader(
                            dataset=test_set,
                            batch_size=batch_size,
                            shuffle=False)

# networks
#----------------------------------------------------------------------

class MyNeuralNetwork1(nn.Module):
    """
    input -> conv2d -> relu -> conv2d -> relu -> maxpooling -> dropout
          -> conv2d -> relu -> conv2d -> relu -> maxpooling -> dropout
          -> flatten -> dense -> relu -> dense -> softmax
    """

    def __init__(self):
        super(MyNeuralNetwork1, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.25)
        
        self.fc1 = nn.Linear(64 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout(self.pool(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.dropout(self.pool(x))

        x = x.view(-1, num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #x = F.softmax(self.fc2(x), dim=1)

        return x

    def name(self):
        return "MyNeuralNetwork1"

class MyNeuralNetwork2(nn.Module):
    """
    LeNet-5

    input -> conv2d -> tanh -> avgpool -> conv2d -> tanh -> avgpool
          -> flatten -> dense -> tanh -> dense -> tanh -> dense -> softmax
    """

    def __init__(self):
        super(MyNeuralNetwork2, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.tanh(self.conv1(x))) # 28*28:->24*24->12*12
        x = self.pool(F.tanh(self.conv2(x))) # 12*12:->8*8->4*4

        x = x.view(-1, num_flat_features(x))
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)

        return x

    def name(self):
        return "MyNeuralNetwork2"

class MyNeuralNetwork3(nn.Module):
    """
    inspired by AlexNet

    input -> conv2d -> relu -> maxpool -> conv2d -> relu -> maxpool
          -> flatten -> dense -> tanh -> dense -> tanh -> dense -> softmax
    """

    def __init__(self):
        super(MyNeuralNetwork3, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)

        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(20 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # 28*28:->24*24->12*12
        x = self.pool(F.relu(self.conv2(x))) # 12*12:->10*10->5*5

        x = x.view(-1, num_flat_features(x))
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)

        return x

    def name(self):
        return "MyNeuralNetwork3"

class MyNeuralNetwork4(nn.Module):
    """
    AlexNet

    input -> conv2d -> relu -> maxpool -> conv2d -> relu -> maxpool
          -> conv2d -> relu -> conv2d -> relu -> conv2d -> relu -> maxpool
          -> flatten -> dense -> tanh -> dense -> tanh -> dense -> softmax
    """

    def __init__(self):
        super(MyNeuralNetwork4, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, 5, padding=2)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv5 = nn.Conv2d(16, 32, 2)

        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(32 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # 28*28:->28*28->14*14
        x = self.pool(F.relu(self.conv2(x))) # 14*14:->14*14->7*7
        x = F.relu(self.conv3(x)) # 7*7:->7*7
        x = F.relu(self.conv4(x)) # 7*7:->7*7
        x = F.relu(self.conv5(x)) # 7*7:->6*6
        x = self.pool(x) # 6*6:->3*3

        x = x.view(-1, num_flat_features(x))
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)

        return x

    def name(self):
        return "MyNeuralNetwork4"

#----------------------------------------------------------------------

def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

#----------------------------------------------------------------------

## training
#model = MyNeuralNetwork1().to(device)
#model = MyNeuralNetwork2().to(device)
#model = MyNeuralNetwork3().to(device)
model = MyNeuralNetwork4().to(device)

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
criterion = nn.CrossEntropyLoss()

train_loss_history = []
test_loss_history = []

train_acc_history = []
test_acc_history = []

best_acc = 0.0
since = time.time()
for epoch in range(num_epochs):
    print("Epoch {}/{}".format(epoch, num_epochs - 1))
    print("-"*10)

    # Each epoch has a training and validation phase
    for phase in ["train", "test"]:
        if phase == "train":
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # iterate over data
        for batch_idx, (inputs, labels) in enumerate(data_loaders[phase]):
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            if batch_idx % 10 == 0:
                print('{} Batch: {} of {}'.format(phase, batch_idx, len(data_loaders[phase])))

        epoch_loss = running_loss / len(data_loaders[phase].dataset)
        epoch_acc = running_corrects.double() / len(data_loaders[phase].dataset)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        # deep copy the model
        if phase == 'test' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        if phase == 'test':
            test_loss_history.append(epoch_loss)
            test_acc_history.append(epoch_acc)
        if phase == 'train':
            train_loss_history.append(epoch_loss)
            train_acc_history.append(epoch_acc)

    print()

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))

plt.title("Validation/Test Loss vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Validation/Test Loss")
plt.plot(range(1, num_epochs+1), train_loss_history, label="Train")
plt.plot(range(1, num_epochs+1), test_loss_history, label="Test")
plt.ylim((0, 1.))
plt.xticks(np.arange(1, num_epochs+1, 1.0))
plt.legend()
plt.show()

acc_train_hist = [h.cpu().numpy() for h in train_acc_history]
acc_test_hist = [h.cpu().numpy() for h in test_acc_history]

plt.title("Validation/Test Accuracy vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Validation/Test Accuracy")
plt.plot(range(1, num_epochs+1), acc_train_hist, label="Train")
plt.plot(range(1, num_epochs+1), acc_test_hist, label="Test")
plt.ylim((0, 1.))
plt.xticks(np.arange(1, num_epochs+1, 1.0))
plt.legend()
plt.show()

examples = enumerate(data_loaders['test'])
batch_idx, (example_data, example_targets) = next(examples)
with torch.no_grad():
  output = model(example_data.to(device))

categories = {
    0:	'T-shirt/top',
    1:	'Trouser',
    2:	'Pullover',
    3:	'Dress',
    4:	'Coat',
    5:	'Sandal',
    6:	'Shirt',
    7:	'Sneaker',
    8:	'Bag',
    9:	'Ankle boot'
}

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Pred: {}".format(
        categories[output.data.max(1, keepdim=True)[1][i].item()]))
    plt.xticks([])
    plt.yticks([])
plt.show()

# inspired by
# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html