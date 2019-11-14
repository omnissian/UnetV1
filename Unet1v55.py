


import torch
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

# import
# import parallelTestModule
#
# if __name__ == '__main__':
#     extractor = parallelTestModule.ParallelExtractor()
#     extractor.runInParallel(numProcesses=2, numThreads=4)


#pixel

random.seed(12)
#C:\Users\user\Desktop\DontTouchPLSpytorch\nets\unet1v1\data_example
# path_learn_in="/storage/MazurM/Task1/ImgData/img/"
# path_learn_in="C:/Users/user/Desktop/DontTouchPLSpytorch/nets/unet1v1/data_example/in/"
# path_learn_label="C:/Users/user/Desktop/DontTouchPLSpytorch/nets/unet1v1/data_example/label/"
path_learn_in="/storage/MazurM/Task1/ImgData/img/"
path_learn_label="/storage/MazurM/Task1/ImgData/mask_build/"
names_learn_in=os.listdir(path_learn_in)
names_learn_label=os.listdir(path_learn_label)
img_paths=[]
# img_paths[:]=path_learn_in+names_learn_in[:]
for i in range (len(names_learn_in)):
    img_paths.append(path_learn_in+names_learn_in[i])

# train_data=[]
# for i in range(len(names_learn_in)):
#     img_in=Image.open(path_learn_in+names_learn_in[i])
#     img_out=Image.open(path_learn_label+names_learn_in[i])
#     train_data.append([torch.from_numpy(np.array(img_in)),torch.from_numpy(np.array(img_out))])

class MyDataset(data.Dataset):
    def __init__(self, type='train'):
        # self.img_paths = []
        self.img_paths = img_paths
        # здесь логика как добавить пути до каждой картинки
    def __getitem__(self, index):
        # img = cv2.imread(self.img_paths[index])
        img=Image.open(path_learn_in+names_learn_in[index])
        label=Image.open(path_learn_label+names_learn_in[index])
        if((random.randint(1,100))>60):
            rotation=random.randint(1,3)
            img=img.rotate((rotation*90),expand=True)
            label=label.rotate((rotation*90),expand=True)
        img=np.asarray(img)
        label=np.asarray(label)
        img_tensor = torch.from_numpy(img)
        label=torch.from_numpy(label)
        #описать логику как достать лейбл конкретно для этой картинки (например вытащить название картинки из пути и по названию найти ее лейбл)
        return img_tensor, label
    def __len__(self):
        return len(self.img_paths)

#--------------hyper parameters------------
batch_size=4
num_workers=0
#--------------hyper parameters------------ # пересобирает батч
def detection_collate(batch):
    targets = []
    imgs = []
    for sample in batch:
        tmpr_l=sample[0]
        tmpr_l=tmpr_l.permute(2,0,1)

        tmpr_r=sample[1]
        tmpr_r=tmpr_r.permute(2,0,1)
        imgs.append(tmpr_l)
        targets.append((tmpr_r))
        # imgs.append(torch.FloatTensor(sample[0]).permute(2, 0, 1))
        # targets.append(torch.FloatTensor([[sample[1]]]))
        # targets.append(torch.FloatTensor(sample[1].permute(2,0,1)))

    return torch.stack(imgs, 0), targets #




train_dataset=MyDataset()
data_train_loader=data.DataLoader(train_dataset, batch_size,num_workers=num_workers,shuffle=True,collate_fn=detection_collate,pin_memory=True,drop_last=True)

# data_example_train=iter(data_train_loader)

# print(data_example_train.next())
print("wtf")

class Model(nn.Module):
    img_channels=3
    # features=64
    def __init__(self):
        super(Model,self).__init__()
        img_channels=3
        features=64
        self.conv2d11=nn.Conv2d(in_channels=img_channels,out_channels=features, kernel_size=3, stride=1,padding=1)
        self.relu1=nn.ReLU()
        self.conv2d12=nn.Conv2d(in_channels=features,out_channels=features, kernel_size=3, stride=1,padding=1) #>>>>
        self.maxPool1=nn.MaxPool2d(kernel_size=2,stride=2) #--------drop ----maxpool---

        self.conv2d21=nn.Conv2d(in_channels=features,out_channels=features*2, kernel_size=3, stride=1,padding=1)
        self.conv2d22 = nn.Conv2d(in_channels=features*2, out_channels=features*2, kernel_size=3, stride=1, padding=1)#>>>
        self.maxPool2=nn.MaxPool2d(kernel_size=2,stride=2) #--------drop ----maxpool---

        self.conv2d31=nn.Conv2d(in_channels=features*2,out_channels=features*4, kernel_size=3, stride=1,padding=1)
        self.conv2d32=nn.Conv2d(in_channels=features*4,out_channels=features*4, kernel_size=3, stride=1,padding=1)#>>
        #after it save out for concat1
        self.maxPool3=nn.MaxPool2d(kernel_size=2,stride=2)#--------drop ----maxpool---

        self.conv2d41=nn.Conv2d(in_channels=features*4,out_channels=features*8, kernel_size=3,stride=1,padding=1)
        self.conv2d42=nn.Conv2d(in_channels=features*8,out_channels=features*8, kernel_size=3,stride=1,padding=1)#--------concat it (4)

        self.maxPool4=nn.MaxPool2d(kernel_size=2, stride=2)#--------drop ----maxpool---
#---------------------------------------------------bottom-----------------------------------

        self.conv2d51=nn.Conv2d(in_channels=features*8,out_channels=features*16, kernel_size=3,stride=1,padding=1)
        self.conv2d52=nn.Conv2d(in_channels=features*16,out_channels=features*16, kernel_size=3,stride=1,padding=1)
#---------------------------------------------------bottom-----------------------------------
        # self.ConvTrans2d1=nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=3) #<
        self.ConvTrans2d1=nn.ConvTranspose2d(in_channels=features*16,out_channels=features*16,kernel_size=2,stride=2,padding=0) #< #--------concat it (4)
        # -------------concat it (4)-----
        # #self.concat1   --will be write in forward section
        self.conv2d61=nn.Conv2d(in_channels=features*16+features*8,out_channels=features*8, kernel_size=3,stride=1,padding=1)
        self.conv2d62=nn.Conv2d(in_channels=features*8,out_channels=features*8, kernel_size=3,stride=1,padding=1)
        # self.maxPool1=nn.MaxPool2d(kerne_size=2,stride=1) #--------drop ----maxpool---
        # self.ConvTrans2d2=nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3,stride=1,padding=3) # <<
        self.ConvTrans2d2=nn.ConvTranspose2d(in_channels=features*8, out_channels=features*8, kernel_size=2,stride=2,padding=0) # <<
        # #self.concat2   --will be write in forward section
        self.conv2d71=nn.Conv2d(in_channels=features*8+features*4,out_channels=features*4, kernel_size=3,stride=1,padding=1)
        self.conv2d72=nn.Conv2d(in_channels=features*4,out_channels=features*4, kernel_size=3,stride=1,padding=1)

        # self.ConvTrans2d3=nn.ConvTranspose2d(in_channels=192, out_channels=64, kernel_size=3,stride=1,padding=3) # <<
        self.ConvTrans2d3=nn.ConvTranspose2d(in_channels=features*4, out_channels=features*4, kernel_size=2,stride=2,padding=0) # <<<
        # #self.concat3   --will be write in forward section
        self.conv2d81=nn.Conv2d(in_channels=features*4+features*2,out_channels=features*2, kernel_size=3,stride=1,padding=1)
        self.conv2d82=nn.Conv2d(in_channels=features*2,out_channels=features*2, kernel_size=3,stride=1,padding=1)
        self.ConvTrans2d4=nn.ConvTranspose2d(in_channels=features*2, out_channels=features*2, kernel_size=2,stride=2,padding=0) # <<<<
        #+
        self.conv2d91 = nn.Conv2d(in_channels=features*2+features*1, out_channels=features*1, kernel_size=3, stride=1, padding=1)
        self.conv2d92 = nn.Conv2d(in_channels=features*1, out_channels=features*1, kernel_size=3, stride=1, padding=1)

        #no activation after this layer
    def forward(self,x):
        print(x.size())
        out=self.conv2d11(x)
        print("out=self.conv2d11(x)")
        print(out.size())
        out=self.relu1(out)
        out=self.conv2d12(out)
        print("out=self.conv2d12(out)")
        print(out.size())
        out=self.relu1(out)
        concat1=out # ----------------------------------1>>>> MAX POOL 1
        print("size concat1", concat1.size())
        out=self.maxPool1(out)
        print("out=self.maxPool1(out)")
        print(out.size())
        out=self.conv2d21(out)
        print("out=self.conv2d21(out)")
        print(out.size())
        out=self.conv2d22(out)
        print("out=self.conv2d22(out)")
        print(out.size())
        concat2=out  # ---------------------------------->>> MAX POOL 2
        print("size concat2", concat2.size())
        out=self.maxPool2(out)
        print("out=self.maxPool2(out)")
        print(out.size())
        out=self.conv2d31(out)
        print("out=self.conv2d31(out)")
        print(out.size())
        out=self.conv2d32(out)
        print("--------------concatenate this in the future!-----------")
        print("out=self.conv2d32(out)")
        print(out.size())
        print("------------------------------------------")
        concat3=out # ---------------------------------->> MAX POOL 3
        print("size concat3", concat3.size())
        # print(concat3.size()) #---------this we will concat 3
        out=self.maxPool3(out)
        print("out=self.maxPool3(out)")
        print(out.size())
        out=self.conv2d41(out)
        print("out=self.conv2d41(out)")
        print(out.size())
        out=self.conv2d42(out)
        print("out=self.conv2d42(out)")
        print(out.size())
        concat4=out # ---------------------------------->> MAX POOL 4 (512+1024 = 1536)
        out=self.maxPool4(out)    # max pool 4
        out=self.conv2d51(out)
        print("out=self.conv2d51(out)")
        print(out.size())
        out=self.conv2d52(out)
        print("out=self.conv2d52(out)")
        print(out.size())
        #------------------------------
        print("--------------concatenate this now!-----------")
        out=self.ConvTrans2d1(out) #---------this we will concatenate with concat4
        print("out=self.ConvTrans2d1(out)")
        print(out.size())
        print("----------------------------------------------")
        print("concatenation!!!")
        print("size concat4: ",concat4.size())
        print("size out: ",out.size())
        out=torch.cat([out,concat4], dim=1)
        print("concat4 size", concat4.size())
        print("out size", out.size())
        out=self.conv2d61(out)
        print("out=self.conv2d61(out)")
        print(out.size())
        out=self.conv2d62(out)
        out=self.ConvTrans2d2(out)
        out=torch.cat([out,concat3],dim=1)
        out=self.conv2d71(out)
        out=self.conv2d72(out)
        # print("size conv 2d72: ", out.size())
        # print("size conv 2d72: ", out.size())
        out=self.ConvTrans2d3(out)
        out=torch.cat([out,concat2],dim=1)
        out=self.conv2d81(out)
        out=self.conv2d82(out)
        out=self.ConvTrans2d4(out)
        out=torch.cat([out,concat1],dim=1)
        out=self.conv2d91(out)
        out=self.conv2d92(out)


net=Model()
print(type(data_train_loader))
# wtf1=enumerate(data_train_loader)
# wtf2=iter(data_train_loader).next()
#
for i, (input_train, targets) in enumerate(data_train_loader):
    input=input_train.float()
    net(input)
    print(i)






print("go1")
net(wtf1.next())
# net.forward()
print("net",net)
# net(data_example_train.next())


print("hello")
