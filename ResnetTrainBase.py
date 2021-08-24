# importing the libraries
import pandas as pd
import numpy as np

# for reading and displaying images
import skimage
from skimage import io
import matplotlib.pyplot as plot
import PIL

# for creating validation set
from sklearn.model_selection import train_test_split

# for evaluating the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# PyTorch libraries and modules
import torch
from torch import nn
from torchvision import models, datasets
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD, lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


#base python libraries
import os
import shutil
import time
import copy



#GlobalVariables
SEED = 5000
DatasetName = "\\APTOS"
Dpath = "C:\\Users\\npnis\\Desktop\\Study\\Assignments And Other Stuff\\Final Year Project" 


#load dataset from drive to local variable
train=pd.read_csv(Dpath+DatasetName+"\\train.csv")
test=pd.read_csv(Dpath+DatasetName+"\\test.csv")

x = train['id_code']
y = train['diagnosis']
train.head()
test.head()


#Image distribution between class labels
#train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.15,stratify=y, random_state=SEED)
#print(train_x.shape, train_y.shape, valid_x.shape, valid_y.shape)
#train_y.hist()
#valid_y.hist()
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class CreateDataset(Dataset):
    def __init__(self, df_data, data_dir = '../input/', transform=None):
        super().__init__()
        self.df = df_data.values
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_name,label = self.df[index]
        #img_path = os.path.join(self.data_dir, img_name+'.jpeg')
        img_path = os.path.join(self.data_dir, str(img_name)+'.png')
        image = io.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)
            #image = image.to(dtype=torch.long)
        return image, label

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
#data_dir = "/content/APTOS"
data_dir = Dpath +"/APTOS"
#data_dir = "/content/drive/MyDrive/Kaggle/train_images"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "inception"

# Number of classes in the dataset
num_classes = 5

# Batch size for training (change depending on how much memory you have)
batch_size = 64

# Number of epochs to train for
num_epochs = 5

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False     

inception = False



#EraseReLU Resnet
import sys
py_file_location = Dpath
sys.path.append(os.path.abspath(py_file_location))
import ResnetEraseReLU as ER

model_ER = ER.resnet50(pretrained=True)
set_parameter_requires_grad(model_ER, feature_extract)
num_ftrs = model_ER.fc.in_features
model_ER.fc = nn.Linear(num_ftrs, num_classes)
input_size = 224
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ER = model_ER.to(device)




# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ER.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ER.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ER.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Setup the loss function
criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
optimizer_ER = SGD(params_to_update, lr=0.001, momentum=0.9)





train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((input_size,input_size)),
    transforms.RandomHorizontalFlip(p=0.4),
    #transforms.ColorJitter(brightness=2, contrast=2),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
#why normalize? replicate preprocessing as on imagenet
#check if can extract green here
test_transforms = transforms.Compose([transforms.Resize(input_size),
                                      transforms.CenterCrop(input_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
train_path = Dpath +"\\APTOS\\train_images"
test_path =  Dpath + "\\APTOS\\test_images"
train_data = CreateDataset(df_data=train, data_dir=train_path, transform=train_transforms)
test_data = CreateDataset(df_data=test, data_dir=test_path, transform=test_transforms)
valid_size = 0.2
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

trainloader = DataLoader(train_data, batch_size=64,sampler=train_sampler)
validloader = DataLoader(train_data, batch_size=64, sampler=valid_sampler)
testloader = DataLoader(test_data, batch_size=64)

#dataloader_dict = {"train":trainloader,"val":validloader,"test":testloader}, when incorporated in test 
dataloader_dict = {"train":trainloader,"val":validloader}
  

#model training without scheduler
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            #for inputs, labels in tqdm(dataloaders[phase]):
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                        #print(loss)
 
                    else:
                        outputs = model(inputs)
                        
                        outputs = outputs.to(dtype = torch.double)
                        loss = criterion(outputs, labels.long())

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            #TBD save model somewhere, maybe drive
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


# Train and evaluate
model_ER, hist = train_model(model_ER, dataloader_dict, criterion, optimizer_ER, num_epochs=num_epochs, is_inception=inception)



#Save the model
#torch.save(model_ER.state_dict(),"/content/drive/MyDrive/APTOS/classifierER.pt")
#torch.save(model_ER.state_dict(),"/content/drive/MyDrive/Kaggle/IclassifierERFT.pt")
torch.save(model_ER.state_dict(),Dpath+"/RclassifierERFT.pt")

class_correct = list(0. for i in range(5))
class_total = list(0. for i in range(5))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classes = ["No DR", "Mild DR","Moderate DR","Severe DR" , "PDR"]

with torch.no_grad():
    for images , labels in validloader:
      

    #for data, dataLabel in testloader:
        #images, labels = testloader
        images = images.to(device)
        labels = labels.to(device)       
        outputs = model_ER(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(5):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))