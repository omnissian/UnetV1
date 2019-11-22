import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
import random
import os
#---import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
import random
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch.utils.data as data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_learn_in="C:/Users/user/Desktop/DontTouchPLSpytorch/Notes/task2/UnetV1/DataSet1/ImgData/img/"
path_learn_targets="C:/Users/user/Desktop/DontTouchPLSpytorch/Notes/task2/UnetV1/DataSet1/ImgData/mask_build/"
path_vaild_in="C:/Users/user/Desktop/DontTouchPLSpytorch/Notes/task2/UnetV1/DataSet1/validation/img/"
path_vaild_targets="C:/Users/user/Desktop/DontTouchPLSpytorch/Notes/task2/UnetV1/DataSet1/validation/mask_build/"
names_learn=os.listdir(path_learn_in)
names_valid=os.listdir(path_vaild_in)

#------------------augmentation----------------------------------------
#----------convert 3Channel colored image to 1channel class tensor--------
#----------Validation targets------------------ create a function of it augmented

list_aug_learn_in = []
list_aug_learn_targets = []



class DataValid(data.Dataset):
    def __init__(self):
        pass
    def __getitem__(self, item):
        img=np.array(Image.open(path_vaild_in+names_valid[item]))
        label=np.array(Image.open(path_vaild_targets+names_valid[item]))
        # img = torch.from_numpy(np.asarray(img))
        label = np.asarray(label[:,:,0]) # [:,:,0] - first channel only because THIS SPECIFIC TASK!!
        height=len(label[:,0])
        width=len(label[0,:])
        for i in range(width):
            for j in range(height):
                if(label[i,j]>=210):
                    label[i,j]=1
                else:
                    label[i,j]=0
        label=torch.from_numpy(label)

        return img, label
    def __len__(self):
        return len(names_valid)

#-----------------------
class DataTrain(data.Dataset):
    def __init__(self):
        pass
    def __getitem__(self, item):
        img=np.array(Image.open(path_learn_in+names_learn[item]))
        label=np.array(Image.open(path_learn_in+names_learn[item]))
        # label=np.array(Image.open(path_learn_in+names_learn[item]))
        # img = torch.from_numpy(np.asarray(img))
        label = np.asarray(label[:,:,0]) # [:,:,0] - first channel only because THIS SPECIFIC TASK!!
        height=len(label[:,0])
        width=len(label[0,:])
        for i in range(width):
            for j in range(height):
                if(label[i,j]>=210):
                    label[i,j]=1
                else:
                    label[i,j]=0
        img=torch.from_numpy(img)
        label=torch.from_numpy(label)
        img=img.permute(2,0,1)
        return img.type(torch.FloatTensor), label.type(torch.FloatTensor)
    def __len__(self):
        return len(names_learn)

#-----------------------



def detection_collate(batch):
    imgs = []
    targets = []
    for sample in batch:
        tmpr_l=sample[0]
        tmpr_l=tmpr_l.permute(2,0,1)
        tmpr_r=sample[1]
        imgs.append(tmpr_l)
        targets.append(tmpr_r)
        #------------my addition for gpu, if code below will not use - unkomment MARK 3 and comment code in the sequence below
        left_return=torch.stack(imgs,0)
        left_return=torch.tensor(left_return,device=device)

        right_return=torch.stack(targets, 1)
        right_return=torch.tensor(right_return,device=device)
        # -----------

    return left_return,right_return

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.img_channels=3
        self.out_channels=1
        features=64
        self.conv2d11=nn.Conv2d(in_channels=self.img_channels,out_channels=self.out_channels,kernel_size=1,stride=1,padding=0)
        self.relu1=nn.ReLU()
    def forward(self,x):
        out=self.conv2d11(x)
        print("out.size() ", out.size())
        out=self.relu1(out)
        return out

batch_size=2
train_set=DataTrain()
data_train=data.DataLoader(train_set,batch_size=batch_size,)
# loss=nn.CrossEntropyLoss()
loss=nn.MSELoss()
learn_rate=0.003
# net=Model().to(device)

net = Model()
net.to(device)


optimizer=torch.optim.SGD(net.parameters(), lr=learn_rate)
for i,(data_in,label) in enumerate(data_train):
    data_in.to(device)
    label.to(device)
    data_in.requires_grad=True
    predict=net(data_in)
    print(predict.size())
    error=loss(predict,label)
    optimizer.zero_grad()
    error.backward()
    optimizer.step()

    print("break point")















