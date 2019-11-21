
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

# cuda=torch.device('cuda')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(torch.cuda.is_available())
# print(device)

# import
# import parallelTestModule
#
# if __name__ == '__main__':
#     extractor = parallelTestModule.ParallelExtractor()
#     extractor.runInParallel(numProcesses=2, numThreads=4)


#pixel

torch.set_default_tensor_type('torch.cuda.FloatTensor')
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




#--------------------------------get item for only 2 classes-----------------------

# image_mask = Image.open(
#             '/storage/UserName/delet_grass_from_build_and_road/db/build/mask_all/' + name_file + '.tiff')
#         image_mask = np.array(image_mask)
#         self.image_mask_1 = image_mask[:,:,0]
#         for j in range(256):
#             for k in range(256):
#                 if image_mask[j,k, 0]!=255 and image_mask[j,k, 1]!=255 and image_mask[j,k, 2]!=255:
#                     self.image_mask_1[j,k]=1.0
#                 else:
#                     self.image_mask_1[j,k]=0.0

#--------------------------------get item for only 2 classes-----------------------

path_valid_in="/storage/MazurM/Task1/validation/img/"
path_valid_label="/storage/MazurM/Task1/validation/mask_build/"

names_valid_in=os.listdir(path_valid_in)
names_valid_label=os.listdir(path_valid_label)
img_valid_paths=[]
for i in range (len(names_valid_in)):
    img_valid_paths.append(path_valid_in+names_valid_in[i])

class ValidationData(data.Dataset):
    def __init__(self, type='train'):
        # self.img_paths = []
        self.img_valid_paths = img_valid_paths
        # здесь логика как добавить пути до каждой картинки

#??????? ?? ?????? - ???????? ??????? (????? ?????)
#
#
#
#
#




    def __getitem__(self, index):
        # img = cv2.imread(self.img_paths[index])
        img = Image.open(path_valid_in + names_valid_in[index])
        label = Image.open(path_valid_label + names_valid_in[index])
        if ((random.randint(1, 100)) > 60):
            rotation = random.randint(1, 3)
            img = img.rotate((rotation * 90), expand=True)
            label = label.rotate((rotation * 90), expand=True)
        img = np.asarray(img)
        label = np.asarray(label)
        self.label_classes = label[:, :, 0]
        height = len(self.label_classes[:, 0])
        width = len(self.label_classes[0, :])
        # label_img_return=np.zeros((height,width), dtype=torch.long) #addition of true bicycle # dtype=torch.float
        label_img_return=torch.ones((height,width),dtype=torch.long,device=device)  # WITH MARK 3
        for i in range(width):
            for j in range(height):
                wtf = self.label_classes[i, j]
                # print("wtf ", wtf)
                if (self.label_classes[i, j] >= 210):
                    # self.label_classes[i, j] = 1
                    label_img_return[i,j]=0
                else:
                    # self.label_classes[i, j] = 0
                    label_img_return[i,j]=1
        img_tensor = torch.from_numpy(img)
        # img_tensor=torch.tensor(img_tensor,device=device)
        # label_img_return=torch.from_numpy((label_img_return)) # WITH MARK 3

        # label_img_return = []
        # label_img_return = torch.from_numpy(self.label_classes)
        # label=torch.from_numpy(label)

        return img_tensor,label_img_return

    # def __getitem__(self, index):
    #     # img = cv2.imread(self.img_paths[index])
    #     img=Image.open(path_valid_in+names_valid_in[index])
    #     label=Image.open(path_valid_label+names_valid_in[index])
    #     if((random.randint(1,100))>60):
    #         rotation=random.randint(1,3)
    #         img=img.rotate((rotation*90),expand=True)
    #         label=label.rotate((rotation*90),expand=True)
    #
    #     img=np.asarray(img)
    #     label=np.asarray(label)
    #     #-----------------label pixel to class label---------bicycle-----
    #     self.label_classes=label[:,:,0]
    #     height=len(self.label_classes[:,0])
    #     width=len(self.label_classes[0,:])
    #     print("type(self.label_classes)= ",type(self.label_classes))
    #     (self.label_classes)[1,0]=1
    #     for i in range(width):
    #         for j in range(height):
    #             wtf=self.label_classes[i,j]
    #             # print("wtf ",wtf)
    #             if (self.label_classes[i,j]>=210):
    #                 self.label_classes[i,j]=1
    #             else:
    #                 self.label_classes[i,j]=0
    #     #-----------------label pixel to class label--------- ???, ??????? ????, ????????? ???????? ?????
    #     # gpi ???? ? ????? - ????????? ???????? (???????? ? ??????? -gpo ? ?? gpo)
    #     # data sheet ???????, ???????? ?????
    #     # ??? ??????????? ?? ??????? ????????????????? -> ???????????? ??? ??????? ?????????????? "??????"
    #     #??? ?????? ????????, ?? ???? ?????? ???????? ??????, ???????? ?????? ?????? - ? ????? ????? ??????? ???????? ?????? ??? ?????????
    #     #??????? ??????? (??????????? ???????? ???????) ??? ??? ???? ???????????? ???????? ????????
    #     #
    #     #
    #     #
    #     img_tensor = torch.from_numpy(img)
    #     label_img_return=[]
    #     label_img_return=torch.from_numpy(self.label_classes)
    #     # label=torch.from_numpy(label)
    #
    #     # img_label=label[:,:,0]
    #     # height=len(img_label[:,0])
    #     # width=len(img_label[0,:])
    #     # for i in range(width):
    #     #     for j in range(height):
    #     #         pix_label=img_label[i,j].item()
    #     #         if(pix_label==255):
    #     #             # print("emplace this pixel by zero on one")
    #     #             img_label[i,j]=1
    #
    #     #описать логику как достать лейбл конкретно для этой картинки (например вытащить название картинки из пути и по названию найти ее лейбл)
    #     return img_tensor,label_img_return
    #     # return img_tensor, label
    def __len__(self):
        return len(self.img_valid_paths)

class MyDataset(data.Dataset):
    def __init__(self, type='train'):
        # self.img_paths = []
        self.img_paths = img_paths #why do we feel it like this? why i should afraid their intentions? who knows... who knows
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
        #? ?????? ???? ???? ?? ????????, ? ??? ?? ?? ??????? ?? ??????? ??????, ??????? ??????????? ????? - ????????? ??????


        #fix this like i fixed for validation data set

        #------------------------------------original pixel to new class--------
        self.label_classes=label[:,:,0]
        height=len(self.label_classes[:,0])
        width=len(self.label_classes[0,:])
        # label_img_return=np.zeros((height,width),dtype=torch.long)
        label_img_return=torch.ones((height,width),dtype=torch.long,device=device) # WITH MARK 2
        for i in range(width):
            for j in range(height):
                wtf=self.label_classes[i,j]
                # print("wtf ",wtf)
                if (self.label_classes[i,j]>=210):
                    # self.label_classes[i,j]=1
                    label_img_return[i,j]=0
                else:
                    # self.label_classes[i,j]=1
                    label_img_return[i,j]=1
        #------------------------------------original pixel to new class--------

        img_tensor = torch.from_numpy(img)
        img_tensor=torch.tensor(img_tensor,device=device)
        # label_img_return=torch.from_numpy((label_img_return)) # WITH MARK 2
        #описать логику как достать лейбл конкретно для этой картинки (например вытащить название картинки из пути и по названию найти ее лейбл)

        return img_tensor, label_img_return

    def __len__(self):
        return len(self.img_paths)

#----------------was checking image dimensions------------deletable----------
# label=Image.open(path_learn_label+names_learn_in[0])
# label=np.asarray(label)
# label=torch.from_numpy(label)
# print("label.size ", label.size())
# print("label[:,0].size() ",label[:,0].size())
# print("-----------------")
# x=label[:,:,0]
# print("x.size() ",x.size())
# print("x[:,0].size() ",x[:,0].size())
# # print("(x[:,0].size()).item() ",(x[:,0].size()).data[0])
# print("len(x[:,0]) ",len(x[:,0]))
# height=len(x[:,0])
# width=len(x[0,:])
# for i in range(height):
#     for j in range(width):
#         wtf=x[i,j].item()
#         # print(wtf)
#         # print("look")
#         if (x[i,j]>=210):
#             x[i,j]=1
#             # print("x[i,j].item() ",x[i,j].item())
#             # print("DAVAI")
#         else:
#             x[i,j]=0
#
#
#
# min=torch.min(x)
# print("min ", min)
# max=torch.max(x)
# print("min ", max)
#
# img=np.array(x)
# plt.imshow(img,cmap='gray')
#
# print("wait for your step")


#----------------was checking image dimensions------------deletable----------




#--------------hyper parameters------------
batch_size=4
num_workers=0
#--------------hyper parameters------------ # пересобирает батч
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
        left_return=torch.tensor(left_return)

        right_return=torch.stack(targets, 1)
        right_return=torch.tensor(right_return)
        # -----------

    return left_return,right_return
    # return torch.stack(imgs,0), torch.stack(targets, 1) # MARK 3

train_dataset=MyDataset()
# train_dataset=train_dataset.to(device) # my addition
data_train_loader=data.DataLoader(train_dataset, batch_size,num_workers=num_workers,shuffle=True,collate_fn=detection_collate, pin_memory=False,drop_last=True)
# data_train_loader=data.DataLoader(train_dataset, batch_size,num_workers=num_workers,shuffle=True,collate_fn=detection_collate,pin_memory=True,drop_last=True)
valid_dataset=ValidationData()
# valid_dataset=valid_dataset.to(device) # my addition
data_valid_loader=data.DataLoader(valid_dataset, batch_size, num_workers=num_workers,shuffle=False, collate_fn=detection_collate, pin_memory=False, drop_last=True)
# data_valid_loader=data.DataLoader(valid_dataset, batch_size, num_workers=num_workers,shuffle=False, collate_fn=detection_collate, pin_memory=True, drop_last=True)


#------------------need to add batch norm, need to add activation layers------------

class Model(nn.Module):
    img_channels=3
    # features=64
    def __init__(self):
        super(Model,self).__init__()
        # Model.to('cuda')
        img_channels=3
        out_channels=1
        features=64
        self.conv2d11=nn.Conv2d(in_channels=img_channels,out_channels=features, kernel_size=3, stride=1,padding=1)
        self.BN1=nn.BatchNorm2d(img_channels)
        self.relu1=nn.ReLU()
        self.conv2d12=nn.Conv2d(in_channels=features,out_channels=features, kernel_size=3, stride=1,padding=1) #>>>>
        self.BN2=nn.BatchNorm2d(features)
        self.maxPool1=nn.MaxPool2d(kernel_size=2,stride=2) #--------drop ----maxpool---

        self.conv2d21=nn.Conv2d(in_channels=features,out_channels=features*2, kernel_size=3, stride=1,padding=1)
        self.BN3=nn.BatchNorm2d(features*2)
        self.conv2d22 = nn.Conv2d(in_channels=features*2, out_channels=features*2, kernel_size=3, stride=1, padding=1)#>>>
        self.BN4=nn.BatchNorm2d(features*2)
        self.maxPool2=nn.MaxPool2d(kernel_size=2,stride=2) #--------drop ----maxpool---

        self.conv2d31=nn.Conv2d(in_channels=features*2,out_channels=features*4, kernel_size=3, stride=1,padding=1)
        self.BN5=nn.BatchNorm2d(features*4)
        self.conv2d32=nn.Conv2d(in_channels=features*4,out_channels=features*4, kernel_size=3, stride=1,padding=1)#>>
        self.BN6=nn.BatchNorm2d(features*4)
        #after it save out for concat1
        self.maxPool3=nn.MaxPool2d(kernel_size=2,stride=2)#--------drop ----maxpool---

        self.conv2d41=nn.Conv2d(in_channels=features*4,out_channels=features*8, kernel_size=3,stride=1,padding=1)
        self.BN7=nn.BatchNorm2d(features*8)
        self.conv2d42=nn.Conv2d(in_channels=features*8,out_channels=features*8, kernel_size=3,stride=1,padding=1)#--------concat it (4)
        self.BN8=nn.BatchNorm2d(features*8)

        self.maxPool4=nn.MaxPool2d(kernel_size=2, stride=2)#--------drop ----maxpool---
#---------------------------------------------------bottom-----------------------------------

        self.conv2d51=nn.Conv2d(in_channels=features*8,out_channels=features*16, kernel_size=3,stride=1,padding=1)
        self.BN9=nn.BatchNorm2d(features*16)
        self.conv2d52=nn.Conv2d(in_channels=features*16,out_channels=features*16, kernel_size=3,stride=1,padding=1)
        self.BN10=nn.BatchNorm2d(features*16)
#---------------------------------------------------bottom-----------------------------------
        # self.ConvTrans2d1=nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=3) #<
        self.ConvTrans2d1=nn.ConvTranspose2d(in_channels=features*16,out_channels=features*16,kernel_size=2,stride=2,padding=0) #< #--------concat it (4)
        # -------------concat it (4)-----
        # #self.concat1   --will be write in forward section
        self.conv2d61=nn.Conv2d(in_channels=features*16+features*8,out_channels=features*8, kernel_size=3,stride=1,padding=1)
        self.BN11=nn.BatchNorm2d(features*8)
        self.conv2d62=nn.Conv2d(in_channels=features*8,out_channels=features*8, kernel_size=3,stride=1,padding=1)
        self.BN12=nn.BatchNorm2d(features*8)
        # self.maxPool1=nn.MaxPool2d(kerne_size=2,stride=1) #--------drop ----maxpool---
        # self.ConvTrans2d2=nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3,stride=1,padding=3) # <<
        self.ConvTrans2d2=nn.ConvTranspose2d(in_channels=features*8, out_channels=features*8, kernel_size=2,stride=2,padding=0) # <<
        # #self.concat2   --will be write in forward section
        self.conv2d71=nn.Conv2d(in_channels=features*8+features*4,out_channels=features*4, kernel_size=3,stride=1,padding=1)
        self.BN13=nn.BatchNorm2d(features*4)
        self.conv2d72=nn.Conv2d(in_channels=features*4,out_channels=features*4, kernel_size=3,stride=1,padding=1)
        self.BN14=nn.BatchNorm2d(features*4)

        # self.ConvTrans2d3=nn.ConvTranspose2d(in_channels=192, out_channels=64, kernel_size=3,stride=1,padding=3) # <<
        self.ConvTrans2d3=nn.ConvTranspose2d(in_channels=features*4, out_channels=features*4, kernel_size=2,stride=2,padding=0) # <<<
        # #self.concat3   --will be write in forward section
        self.conv2d81=nn.Conv2d(in_channels=features*4+features*2,out_channels=features*2, kernel_size=3,stride=1,padding=1)
        self.BN15=nn.BatchNorm2d(features*2)
        self.conv2d82=nn.Conv2d(in_channels=features*2,out_channels=features*2, kernel_size=3,stride=1,padding=1)
        self.BN16=nn.BatchNorm2d(features*2)
        self.ConvTrans2d4=nn.ConvTranspose2d(in_channels=features*2, out_channels=features*2, kernel_size=2,stride=2,padding=0) # <<<<
        #+
        self.conv2d91 = nn.Conv2d(in_channels=features*2+features*1, out_channels=features*1, kernel_size=3, stride=1, padding=1)
        self.BN17=nn.BatchNorm2d(features*2)
        self.conv2d92 = nn.Conv2d(in_channels=features*1, out_channels=features*1, kernel_size=3, stride=1, padding=1)
        self.BN18=nn.BatchNorm2d(features*2)

        # self.conv2d101=nn.Conv2d(in_channels=features*1, out_channels=img_channels,kernel_size=1,stride=1,padding=0) #last conv layer
        self.conv2d101=nn.Conv2d(in_channels=features*1, out_channels=out_channels,kernel_size=1,stride=1,padding=0) #last conv layer



        #no activation after this layer
    def forward(self,x):
        print("type(x) ", type(x))
        # x
        # x=torch.tensor(x,device=device)
        # x.to(device)
        # print(x.size())
        out=self.conv2d11(x)
        # print("out=self.conv2d11(x)")
        # print(out.size())
        out=self.relu1(out) #<< RELU
        out=self.conv2d12(out)
        # print("out=self.conv2d12(out)")
        # print(out.size())
        out=self.relu1(out) #<< RELU
        concat1=out # ----------------------------------1>>>> MAX POOL 1
        # print("size concat1", concat1.size())
        out=self.maxPool1(out)
        # print("out=self.maxPool1(out)")
        # print(out.size())
        out=self.conv2d21(out)
        # print("out=self.conv2d21(out)")
        # print(out.size())
        out=self.relu1(out) #<< RELU
        out=self.conv2d22(out)
        # print("out=self.conv2d22(out)")
        # print(out.size())
        concat2=out  # ---------------------------------->>> MAX POOL 2
        # print("size concat2", concat2.size())
        out=self.relu1(out) #<< RELU
        out=self.maxPool2(out)
        # print("out=self.maxPool2(out)")
        # print(out.size())
        out=self.conv2d31(out)
        out=self.relu1(out) #<< RELU
        # print("out=self.conv2d31(out)")
        # print(out.size())
        out=self.conv2d32(out)
        out=self.relu1(out) #<< RELU
        # print("--------------concatenate this in the future!-----------")
        # print("out=self.conv2d32(out)")
        # print(out.size())
        # print("------------------------------------------")
        concat3=out # ---------------------------------->> MAX POOL 3
        # print("size concat3", concat3.size())
        # print(concat3.size()) #---------this we will concat 3
        out=self.maxPool3(out)
        # print("out=self.maxPool3(out)")
        # print(out.size())
        out=self.conv2d41(out)
        out=self.relu1(out) #<< RELU
        # print("out=self.conv2d41(out)")
        # print(out.size())
        out=self.conv2d42(out)
        out=self.relu1(out) #<< RELU
        # print("out=self.conv2d42(out)")
        # print(out.size())
        concat4=out # ---------------------------------->> MAX POOL 4 (512+1024 = 1536)
        out=self.maxPool4(out)    # max pool 4
        out=self.conv2d51(out)
        out=self.relu1(out) #<< RELU
        # print("out=self.conv2d51(out)")
        # print(out.size())
        out=self.conv2d52(out)
        out=self.relu1(out) #<< RELU
        # print("out=self.conv2d52(out)")
        # print(out.size())
        #------------------------------
        # print("--------------concatenate this now!-----------")
        out=self.ConvTrans2d1(out) #---------this we will concatenate with concat4
        # print("out=self.ConvTrans2d1(out)")
        # print(out.size())
        # print("----------------------------------------------")
        # print("concatenation!!!")
        # print("size concat4: ",concat4.size())
        # print("size out: ",out.size())
        out=torch.cat([out,concat4], dim=1)
        # print("concat4 size", concat4.size())
        # print("out size", out.size())
        out=self.conv2d61(out)
        out=self.relu1(out) #<< RELU
        # print("out=self.conv2d61(out)")
        # print(out.size())
        out=self.conv2d62(out)
        out=self.relu1(out) #<< RELU
        out=self.ConvTrans2d2(out)
        out=torch.cat([out,concat3],dim=1)
        out=self.conv2d71(out)
        out=self.relu1(out) #<< RELU
        out=self.conv2d72(out)
        out=self.relu1(out) #<< RELU
        # print("size conv 2d72: ", out.size())
        # print("size conv 2d72: ", out.size())
        out=self.ConvTrans2d3(out)
        out=torch.cat([out,concat2],dim=1)
        out=self.conv2d81(out)
        out=self.relu1(out) #<< RELU
        out=self.conv2d82(out)
        out=self.relu1(out) #<< RELU
        out=self.ConvTrans2d4(out)
        out=torch.cat([out,concat1],dim=1)
        out=self.conv2d91(out)
        out=self.relu1(out) #<< RELU
        out=self.conv2d92(out)
        out=self.relu1(out) #<< RELU

        out=self.conv2d101(out)

        out=torch.tensor(out) #next addition
        print("OUTPUT SIZE from the NetWork ",out.size())
        # print("here we go")
        return out

    def load_weights(self, weights_file):
        other, ext = os.path.splitext(weights_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(weights_file, map_location=lambda storage, loc: storage))
            #stirct false\true - we could load parameters from certain place
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

net=Model().to(device)
# net=net.to('cuda')
print(type(data_train_loader))
learn_val=3
learn_val=float(learn_val)
learn_rate=learn_val*10**(-3)
optimizer=torch.optim.SGD(net.parameters(), lr=learn_rate)
cost_func=nn.CrossEntropyLoss()
#-------------sequence for save parameters---------

print("please in 1 for save parameters, 0 for pass")
# answer=int(input())
answer=1
save_folder="/storage/MazurM/Task1/SavedParameters/UnetV1/polygon/"
file_saved_parameters="Unet1v1Parameters.pth"

if(answer):
    torch.save(net.state_dict(), os.path.join(save_folder, file_saved_parameters))
else:
    print("do you want to load the weights of net?")
    print("'1' for YES OR  '0' for NO")
    answer = int(input())
    if(answer):
        net.load_weights(save_folder + file_saved_parameters)
    else:pass
#-------------sequence for save parameters---------
print("do you want to train the neural net?")
print("'1' for Train and '0' for not to train the Unetv1")
# answer=int(input())
answer=1

#---------!!!!!!!!!!!!!!!!!!!!!!!!!!----found number of all classes-----------------
# original_pixels=[]
# adds=0
# #---test!!! adds=1
# adds=1
#
#
# #---test!!! adds=1
# for iter_valid_data, (input_valid, targets_valid) in enumerate(data_valid_loader):
#
#     original_pixels=(targets_valid.permute(1,0,2,3)).data[0][:,0,0] # problem!!! DOESNT MATCH!! Because label is one channel, THe channels IN not compatible with OUT channels
#     # and should check each object
#
#
#
#     # ---test!!! adds=1
#     test_cat=original_pixels
#     # test_cat=torch.tensor([0,0,0],device='cpu')
#     print("test_cat.size() ",test_cat.size())
#     original_pixels=torch.stack([original_pixels,original_pixels],dim=0)
#     print("original_pixels.size() ",original_pixels.size())
#     # original_pixels=torch.cat([original_pixels.data[0],test_cat],dim=0)
#     # torch.cat([original_pixels,original_pixels],dim=0]
#     print("original_pixels.size() ",original_pixels.size())
#     print("len(original_pixels) ",len(original_pixels))
#     # ---test!!! adds=1
#     tmp = (targets_valid.permute(1, 0, 2, 3)).data[0]
#     channels = targets_valid.permute(1, 0, 2, 3).size()
#     width = channels[3]
#     height = channels[2]
#     channels = channels[1]
#     # wtf1=targets_valid.permute(1,0,2,3)
#     print("targets_valid.size() ",targets_valid.size())
#     for i in range(batch_size):
#         # original_pix=np.zeros((width,height,channels))
#         # original_pixels=torch.tensor([0,0,0]) # for rgb
#         for i in range(width):
#             for j in range(height):
#                 pix=tmp.data[:,i,j]
#                 WTF_len=len(original_pixels)
#                 if(adds):
#                     is_in=False
#                     for ip in range (len(original_pixels)):
#                         # dict_cur_pix=original_pixels.data[0]
#                         left=original_pixels.data[ip]
#                         right=pix.data
#                         print(right)
#                         print(left)
#                         wtf_logic = (torch.all(torch.eq(pix, original_pixels.data[ip]))).item()
#                         if(wtf_logic):
#                             is_in=True
#                             break
#                         # else:
#                         #     right=right.unsqueeze(0)
#                         #     print("left= ",left)
#                         #     print("right= ",right)
#                         #
#                         #     # original_pixels=torch.cat([original_pixels,pix],dim=0)
#                         #     original_pixels=torch.cat([original_pixels,right],dim=0)
#                         #     break
#                     if(not is_in):
#                         new_pix=pix.unsqueeze(0)
#                         original_pixels=torch.cat([original_pixels,new_pix], dim=0)
#                         pass
#
#                 else:
#                     wtf_logic=(torch.all(torch.eq(pix,original_pixels))).item()
#                     if(wtf_logic):
#                         pass
#                     else:
#                         torch.stack([original_pixels,pix],dim=0)
#                         adds+=1;
#
#                     # wtf_logic=(torch.all(torch.eq(pix,original_pixels))).data[0]
#
#                 # print("original_pixels.size() ",original_pixels.size())
#                 # print("original_pixels[0].size() ",original_pixels[0].size())
#                 # torch.eq(pix,)
#     print("stop")
#             # pixel_ch = 0
#             # img_out=tmp.data[ch]
#             # width,height =img_out.size()
#             # for i in range(width):
#             #     for j in range (height):
#             #         img_out
#         #----------------------------------------------------
#         # for ch in range(channels):
#         #     pixel_ch = 0
#         #     img_out=tmp.data[ch]
#         #     width,height =img_out.size()
#         #     for i in range(width):
#         #         for j in range (height):
#         #             img_out

# -------------found number of all classes-----------------
#---------!!!!!!!!!!!!!!!!!!!!!!!!!!----found number of all classes-----------------


# loss
# n2 loss
# l2,-simple, glubin-next

print("check all stuff")
if (answer):
    print("please in the number>0 of train epochs")
    # epochs=int(input())
    epochs=9
    for num_epoch in range(epochs):
        if(epochs>0):
            predicted=0
            total=0
            with torch.no_grad():
                # wtf_e=enumerate(data_valid_loader)
                for iter_valid_data,(input_valid, targets_valid) in enumerate(data_valid_loader):
                    # input_valid=input_valid.to(device) # cuda() #----------to CUDA--------GPU---------
                    input_valid=input_valid.to(device=device, dtype=torch.long) # dtype=torch.long
                    targets_valid=targets_valid.to(device=device, dtype=torch.long) # dtype=torch.long
                    output=net(input_valid) # original -----here some problem---------------
                    targets_valid=targets_valid.permute(1,0,2) # --------should becommented with next Mark 124
                    size_labels=len(targets_valid)
                    #---------check output in VALID not learn-------------
                    print("-----------VALIDATION - not learn-----------")
                    print("targets_valid.size() ",targets_valid.size()) # torch.Size([4, 256, 256])
                    print("output.size() ",output.size()) # torch.Size([4, 1, 256, 256])
                    print("break point")
                    #---------check output in VALID not learn-------------

                    for x in range(size_labels): # MARK 124 - if commented, this sequence should be commented too
                        total+=1
                        if(torch.equal(output.data[x],targets_valid.data[x])):
                            predicted+=1
                    accuracy=(predicted/total)*100.0
                    print("After epoch ", num_epoch,"Accuracy= ", accuracy)
        print("---->train epoch started<------")

        for i, (input_train, targets) in enumerate(data_train_loader):

            input_train=input_train.to(device=device, dtype=torch.long) #torch.float
            targets=targets.permute(1,0,2) # compatible to validation target # --------should becommented with next Mark 124
            targets=targets.to(device=device, dtype=torch.long) # toorch.float
            input_train = input_train.requires_grad_()
            optimizer.zero_grad()
            predict = net(input_train)
            predict=predict.permute(1,0,2,3)
            # ---------check output in TRAIN  LEARN-------------  (batch size, channels, height, width).
            print("-----------LEARNING  - NET TRAIN-----------")
            print("targets.size() ", targets.size()) # torch.Size([4, 256, 256])
            print("predict.size() ", predict.size()) # torch.Size([1, 4, 256, 256])
            predict_to_loss=targets.unsqueeze(0)
            print("predict_to_loss.size() ",predict_to_loss.size())
            predict=predict.permute(1,0,2,3)
            predict_to_loss=predict_to_loss.permute(1,0,2,3)
            predict_to_loss=predict_to_loss.float()
            targets=targets.float()
            print("break point")
            # ---------check output in VALID not learn-------------
            loss = cost_func(predict_to_loss, targets)
            # loss = cost_func(predict, targets) # ValueError: Expected input batch_size (1) to match target batch_size (4).
            loss.backward()
            optimizer.step()
        print("training Epoch ?",num_epoch," ended")
    print("All training epochs are passed")
else:pass # not to train


print("go1")
net(wtf1.next())
# net.forward()
print("net",net)
# net(data_example_train.next())


print("hello")



