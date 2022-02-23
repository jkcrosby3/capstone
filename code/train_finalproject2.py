from __future__ import print_function, division

import os
import os.path
from _csv import reader
from image_folder import make_dataset
import pandas as pd
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from PIL import Image
import random
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from torch.autograd import Variable
# from torchvision import transforms, utils
import torchvision.transforms as transforms
from torchvision import models
import warnings

warnings.filterwarnings("ignore")
plt.ion()

#"D:\NOAA-data\AWS_project\FinalProject\data_subset"
data_dir = os.listdir("D:\NOAA-data\AWS_project\FinalProject\data_subset") # './data_random/'
data_dir2 =  os.listdir("D:\NOAA-data\AWS_project\FinalProject\data_subset_2") # './data_subset_2/'


class ImgDataset():
    def __init__(self, dataroot, csv, batch_size=40):
        self.batchSize = batch_size
        self.root = dataroot
        self.frame = pd.read_csv(csv, header=None)
        self.image_paths = sorted(make_dataset(self.root))
        self.dataset_size = len(self.image_paths)
      
    def __getitem__(self, index):        
        # ## input A (orig maps)
        i_path = self.image_paths[index]     
        img = Image.open(i_path).convert('RGB')        
        transform_img = self.get_transform()
        img_tensor = transform_img(img)

        df = self.frame
        
        image_name = i_path.split('/')[-1]
        label_string = df.loc[df[0] == image_name, 1].iloc[0]
        if label_string == 'not_interesting':
            label = 1
        else:
            label = 0
        input_dict = {'img': img_tensor, 'label': label}

        return input_dict

    def __len__(self):
        return len(self.image_paths) // self.batchSize * self.batchSize

    def get_transform(self, method=Image.BICUBIC, normalize=True, loadsize=512):
        transform_list = []
        osize = [loadsize, loadsize]
        transform_list.append(transforms.Scale(osize, method))
        transform_list += [transforms.ToTessor()]
        if normalize:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        return transforms.Compose(transform_list)
# -----------end class class ImgDataset()-------------


# %% --------------------------------------- Set-Up --------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# use_gpu = torch.cuda.is_available()
SEED_T = torch.manual_seed(42)
SEED = np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ---------------------------------------------Model-------------------------------------
# from torchvision import models

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = models.vgg16(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        num_features = self.model.classifier[6].in_features
        features = list(self.model.classifier.children())[:-1]  # Remove last layer
        features.extend([nn.Linear(num_features, 1)])  # Add our layer with 1 output
        self.model.classifier = nn.Sequential(*features)  # Replace the model classifier
        # print(self.model)

    def forward(self, orig_image):
        return self.model(orig_image)
# ---------- end class CNN() ----------


continue_train = False

# %% -------------------------------------- Training Loop ----------------------------------------------------------
print("Starting training loop...")

LR = 1e-5
N_EPOCHS = 30

if continue_train is False:
    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
else:
    model = CNN().to(device)
    checkpoint = torch.load('checkpoint_vg16.pth')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=LR)
    optimizer.load_state_dict(checkpoint['optimizer'])

criterion = nn.BCEWithLogitsLoss()

dataset = ImgDataset(data_dir, data_dir2 + 'labels.csv')
validation_split = .2
shuffle_dataset = True
random_seed = 42
batch_size = 16

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
loss_values = []
for epoch in range(N_EPOCHS):
    running_loss = 0.0
    for batchidx, image_dict in enumerate(train_loader):

        optimizer.zero_grad()

        # Generate predictions
        image = Variable(image_dict['img'].data.cuda())
        out = model(image)
        target = image_dict['label'].unsqueeze(1)
        target = target.type_as(out)
        # print(out,target)

        # Calculate loss
        loss = criterion(out, target)
        # Backpropagation
        loss.backward()
        running_loss += loss.item() * batch_size
        # Update model parameters
        optimizer.step()
        if epoch % 2 == 0:
            checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, 'checkpoint_vg16.pth')
        # spacing was off in the next 2 lines, was 4 and 2 respectively, should it therefore be 2 tabs, and one tab?
        print("Train epoch {}, {}/{} Loss:{:.6f}".format(epoch, batchidx * len(image_dict['img']),
                                                         len(train_loader.dataset), loss.item()))
    loss_values.append(running_loss / len(train_indices))

plt.plot(loss_values)


# %% -------------------------------------- Testing the model ----------------------------------------------------------

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc


model = CNN().to(device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
output_all = []
target_all = []
with torch.no_grad():
    for batchidx, image_dict in enumerate(validation_loader):
        optimizer.zero_grad()
        # Generate predictions\
        image = Variable(image_dict['img'].data.cuda())
        out = model(image)

        target = image_dict['label'].unsqueeze(1)
        target = target.type_as(out)
        # print(out,target)

        # Calculate test loss
        loss_test = criterion(out, target)

        output_all.extend(out)
        target_all.extend(target)
        print('Validation Loss {:.5f}'.format(loss_test))
        # validation_output = np.where(out.cpu().numpy()>0.5,1,0)
    
output_all = torch.tensor(output_all)
target_all = torch.tensor(target_all)

acc = binary_acc(output_all, target_all)
print('Validation Accuracy:', acc.cpu().numpy())
