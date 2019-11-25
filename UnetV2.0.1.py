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
        img=torch.from_numpy(img)
        img=img.permute(2,0,1)

        # return img, label.type()
        return img.type(torch.FloatTensor), label.type(torch.FloatTensor)
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
        self.out_channels=1 #depends on cost function, and type(dimensions) of labels too
        self.features=64
        features=64
        #the two layers below is used ony for test net on CPU
        self.conv2dtest=nn.Conv2d(in_channels=self.img_channels,out_channels=self.out_channels,kernel_size=3, stride=1,padding=1)
        self.BNtest=nn.BatchNorm2d(self.out_channels)


        self.conv2d11=nn.Conv2d(in_channels=self.img_channels,out_channels=self.features,kernel_size=3,stride=1,padding=1)
        self.relu1=nn.ReLU()
        self.BN1=nn.BatchNorm2d(self.features)
        self.conv2d12=nn.Conv2d(in_channels=self.features,out_channels=self.features,kernel_size=3,stride=1,padding=1)
        self.maxPool1=nn.MaxPool2d(kernel_size=2,stride=2) #---drop out---

        self.conv2d21=nn.Conv2d(in_channels=self.features,out_channels=self.features*2,kernel_size=3,stride=1,padding=1)
        self.BN2=nn.BatchNorm2d(self.features*2)
        self.conv2d22=nn.Conv2d(in_channels=self.features*2,out_channels=self.features*2,kernel_size=3,stride=1,padding=1)
        self.maxPool2=nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv2d31=nn.Conv2d(in_channels=self.features*2,out_channels=self.features*4,kernel_size=3,stride=1,padding=1)
        self.BN3=nn.BatchNorm2d(self.features*4)
        self.conv2d31=nn.Conv2d(in_channels=self.features*4,out_channels=self.features*4,kernel_size=3,stride=1,padding=1)
        self.maxPool3=nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv2d41=nn.Conv2d(in_channels=self.features*4,out_channels=self.features*8,kernel_size=3,stride=1,padding=1)
        self.conv2d42=nn.Conv2d(in_channels=self.features*8,out_channels=self.features*8,kernel_size=3,stride=1,padding=1)
        self.BN4=nn.BatchNorm2d(self.features*8)
        self.maxPool4=nn.MaxPool2d(kernel_size=2,stride=2)

        #------bottom-------------bottle neck-----------
        self.conv2d51 = nn.Conv2d(in_channels=self.features * 8, out_channels=self.features * 16, kernel_size=3, stride=1, padding=1)
        self.conv2d52 = nn.Conv2d(in_channels=self.features * 16, out_channels=self.features * 16, kernel_size=3, stride=1, padding=1)
        self.BN5=nn.BatchNorm2d(self.features*16)
        #------bottom-------------bottle neck-----------
        self.TransConv2d1=nn.ConvTranspose2d(in_channels=self.features*16,out_channels=self.features*16,kernel_size=2,stride=2,padding=0)
        self.conv2d61 = nn.Conv2d(in_channels=self.features * 16+self.features+8, out_channels=self.features * 8, kernel_size=3, stride=1, padding=1)
        self.BN6 = nn.BatchNorm2d(self.features * 8)
        self.conv2d62 = nn.Conv2d(in_channels=self.features * 8, out_channels=self.features * 8, kernel_size=3, stride=1, padding=1)

        self.TransConv2d2=nn.ConvTranspose2d(in_channels=self.features*8,out_channels=self.features*8,kernel_size=2,stride=2,padding=0)
        self.BN7=nn.BatchNorm2d(self.features*4)
        self.conv2d71 = nn.Conv2d(in_channels=self.features * 8+self.features*4, out_channels=self.features * 4, kernel_size=3, stride=1, padding=1)
        self.conv2d72 = nn.Conv2d(in_channels=self.features * 4, out_channels=self.features * 4, kernel_size=3, stride=1, padding=1)

        self.TransConv2d3=nn.ConvTranspose2d(in_channels=self.features*16,out_channels=self.features*16,kernel_size=2,stride=2,padding=0)
        self.conv2d81 = nn.Conv2d(in_channels=self.features * 4+self.features*2, out_channels=self.features * 2, kernel_size=3, stride=1, padding=1)
        self.BN8=nn.BatchNorm2d(self.features*2)
        self.conv2d82 = nn.Conv2d(in_channels=self.features * 2, out_channels=self.features * 2, kernel_size=3, stride=1, padding=1)

        self.TransConv2d4=nn.ConvTranspose2d(in_channels=self.features*2,out_channels=self.features*2,kernel_size=2,stride=2,padding=0)
        self.conv2d91 = nn.Conv2d(in_channels=self.features * 2+self.features*1, out_channels=self.features * 1, kernel_size=3, stride=1, padding=1)
        self.BN9=nn.BatchNorm2d(self.features*1)
        self.conv2d92 = nn.Conv2d(in_channels=self.features * 1, out_channels=self.features * 1, kernel_size=3, stride=1, padding=1)

        self.conv2d101=nn.Conv2d(in_channels=self.features * 1, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        out=self.conv2dtest(x)
        out=self.BNtest(out)
        out=self.relu1(out)
        return out

    # def forward(self,x):
    #     out=self.conv2d11(x)
    #     out=self.BN1(out)
    #     out=self.relu1(out)
    #     out=self.conv2d12(out)
    #     out=self.BN1(out)
    #     out=self.relu1(out)
    #     concat1=out
    #     out=self.maxPool1(out)
    #     #section  #1 Above
    #     out=self.conv2d21(x)
    #     out=self.BN2(out)
    #     out=self.relu1(out)
    #     out=self.conv2d22(out)
    #     out=self.BN2(out)
    #     out=self.relu1(out)
    #     concat2=out
    #     out=self.maxPool2(out)
    #     #section  #2 Above
    #     out=self.conv2d31(x)
    #     out=self.BN3(out)
    #     out=self.relu1(out)
    #     out=self.conv2d32(out)
    #     out=self.BN3(out)
    #     out=self.relu1(out)
    #     concat3=out
    #     out=self.maxPool3(out)
    #     #section  #3 Above
    #     out=self.conv2d41(x)
    #     out=self.BN4(out)
    #     out=self.relu1(out)
    #     out=self.conv2d42(out)
    #     out=self.BN4(out)
    #     out=self.relu1(out)
    #     concat4=out
    #     out=self.maxPool4(out)
    #     #section  #4 Above
    #
    #     out=self.conv2d51(x)
    #     out=self.BN5(out)
    #     out=self.relu1(out)
    #     out=self.conv2d52(out)
    #     out=self.BN5(out)
    #     out=self.relu1(out)
    #     concat5=out
    #     out=self.maxPool5(out)
    #     #section  #5 Above bottle neck
    #
    #     out=self.TransConv2d1(out)
    #     out=torch.cat([out,concat4],dim=1) #probably too depends on cost function and label type represent at wich dimension should we stack
    #     out=self.conv2d61(out)
    #     out=self.BN6(out)
    #     out=self.relu1(out)
    #     out=self.conv2d62(out)
    #     out=self.BN6(out)
    #     out=self.relu1(out)
    #     #seciton UpSampling Associtated with #4 above
    #
    #     out=self.TransConv2d2(out)
    #     out=torch.cat([out,concat3],dim=1) #probably too depends on cost function and label type represent at wich dimension should we stack
    #     out=self.conv2d71(out)
    #     out=self.BN7(out)
    #     out=self.relu1(out)
    #     out=self.conv2d72(out)
    #     out=self.BN7(out)
    #     out=self.relu1(out)
    #     #seciton UpSampling Associtated with #3 above
    #
    #     out=self.TransConv2d3(out)
    #     out=torch.cat([out,concat2],dim=1) #probably too depends on cost function and label type represent at wich dimension should we stack
    #     out=self.conv2d81(out)
    #     out=self.BN8(out)
    #     out=self.relu1(out)
    #     out=self.conv2d82(out)
    #     out=self.BN8(out)
    #     out=self.relu1(out)
    #     #seciton UpSampling Associtated with #2 above
    #
    #     out = self.TransConv2d4(out)
    #     out = torch.cat([out, concat1], dim=1)  # probably too depends on cost function and label type represent at wich dimension should we stack
    #     out = self.conv2d91(out)
    #     out = self.BN9(out)
    #     out = self.relu1(out)
    #     out = self.conv2d92(out)
    #     out = self.BN9(out)
    #     out = self.relu1(out)
    #     # seciton UpSampling Associtated with #1 above
    #
    #     out=self.conv2d101(out)
    #
    #     # out=torch.tensor(out)
    #     return out

batch_size=2
train_set=DataTrain()
data_train=data.DataLoader(train_set,batch_size=batch_size,) # train data

batch_size_valid=4
valid_data=DataValid()
data_valid=data.DataLoader(valid_data,batch_size=batch_size_valid)
# loss=nn.CrossEntropyLoss()
loss=nn.MSELoss()
learn_rate=0.003
# net=Model().to(device)

net = Model()
net.to(device)

plt.ion

optimizer=torch.optim.SGD(net.parameters(), lr=learn_rate)

amount_epochs=40
for epoch in range(amount_epochs):
    with torch.no_grad():
        for i,(data_in, label) in enumerate(data_valid):
            data_in.to(device)
            label.to(device)
            print("data_in.size() ",data_in.size())
            predict=net(data_in) # RuntimeError: Expected 4-dimensional input for 4-dimensional weight 1 3 3 3, but got 3-dimensional input of size [3, 256, 256] instead
            print("predict.size() ",predict.size())
            print("label.size() ",label.size())
            out_img=predict.numpy()[0,0,:,:]
            label_img=label.numpy()[0,:,:]
            #-------------------- cv2.imwrite (path save, numpy array)
            #plt.savefig
            #--------------------
            plt.imshow(out_img)
            plt.imshow(label_img)
            plt.close()
            # error=loss(predict,label)
            print("predict[:,0,:,:].size() ", predict[:,0,:,:].size())
            error=loss(predict[:,0,:,:],label)
            print(type(error))
            print (error.item())
            predicted=(label==predict[:,0,:,:]).sum() #predicted type - tensor
            print("predicted ",predicted)
            print("break point")



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















