from PIL import Image
import matplotlib as plt # pyplot!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision as tv
import os
import random
random.seed(777)
import torch.nn as nn

# im = Image.open("/storage/MazurM/Task1/ImgData/img/205_0_0.tiff")
# names_learn_in=os.listdir("/storage/MazurM/Task1/ImgData/img/")
#------------ccreate train data set---------------------
path_learn_in="/storage/MazurM/Task1/ImgData/img/"
path_learn_out="/storage/MazurM/Task1/ImgData/mask_build/"
names_learn_in=os.listdir(path_learn_in)
train_data=[]
tuple1=tuple()
for i in range(len(names_learn_in)):
    img_in=Image.open(path_learn_in+names_learn_in[i])
    img_out=Image.open(path_learn_out+names_learn_in[i])
    train_data.append((np.array(img_in),np.array(img_out)))
    if(random.randint(1,100)>60):
        rotation=random.randint(1,3)
        train_data.append((img_in.rotate(rotation*90,expand=True),img_out.rotate(rotation*90,expand=True)))
#------------creat validation data set-------------------------
path_valid_in="/storage/MazurM/Task1/validation/img/"
path_valid_out="/storage/MazurM/Task1/validation/mask_build/"
names_valid_in=os.listdir(path_valid_in)
test_data=[]
for i in range(len(names_valid_in)):
    img_in=Image.open(path_valid_in+names_valid_in[i])
    img_out=Image.open(path_valid_out+names_valid_in[i])
    test_data.append((np.array(img_in),np.array(img_out)))
    if(random.randint(1,100)>50):
        rotation=random.randint(1,3)
        test_data.append((img_in.rotate((rotation*90),expand=True),img_out.rotate(rotation*90,expand=True)))
#-----------------network-------------------
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv2d11=nn.Conv2d(in_channels=3,out_channels=32, kernel_size=3, stride=1,padding=3)
        self.relu1=nn.ReLU()
        self.conv2d12=nn.Conv2d(in_channels=32,out_channels=64, kernel_size=3, stride=1,padding=3) #>>>
        self.maxPool1=nn.MaxPool2d(kerne_size=2,stride=1) #--------drop ----maxpool---

        self.conv2d21=nn.Conv2d(in_channels=64,out_channels=64, kernel_size=3, stride=1,padding=3)
        self.conv2d22 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=3)#>>
        self.maxPool2=nn.MaxPool2d(kerne_size=2,stride=1) #--------drop ----maxpool---

        self.conv2d31=nn.Conv2d(in_channels=128,out_channels=128, kernel_size=3, stride=1,padding=3)
        self.conv2d32=nn.Conv2d(in_channels=128,out_channels=256, kernel_size=3, stride=1,padding=3)#>
        self.maxPool3=nn.MaxPool2d(kernel_size=2,stride=1)#--------drop ----maxpool---

        self.conv2d41=nn.Conv2d(in_channels=256, kernel_size=3,stride=1,padding=3)
        self.conv2d42=nn.Conv2d(in_channels=256, kernel_size=3,stride=1,padding=3)

        self.concat1=nn.


#-----------------------
        self.conv2d2=nn.Conv2d(in_channels=3,out_channels=64, kernel_size=3, stride=1,padding=3) #>
        self.maxPool1=nn.MaxPool2d(kerne_size=2,stride=1) #--------drop ----maxpool---




print("all is done")

# for i in range(len(names_learn_in)):
#     with open(path_learn+names_learn_in[i]) as in_img:
#
#         wtf=Image.open(in_img)
#         wtf2=np.array(wtf)
#         learn_in.append(np.array(Image.open(in_img)))
# print("delta")



print(type(im))
x=np.array(im)
print(x)
print(type(x))
print(x.shape)
plt.imshow(x)
plt.show()

# im.show()
# x.show()
# im.rotate(45).show()
