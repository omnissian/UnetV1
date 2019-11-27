import pdb
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
#import matplotlib.pyplot as plt
import random
import os
#---import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
#import matplotlib.pyplot as plt
import random
import os
#import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch.utils.data as data
import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_learn_in="/home/std_11/MazurM/UnetV1/DataSet1/ImgData/img/"
path_learn_targets="/home/std_11/MazurM/UnetV1/DataSet1/ImgData/mask_build/"
path_vaild_in="/home/std_11/MazurM/UnetV1/DataSet1/validation/img/"
path_vaild_targets="/home/std_11/MazurM/UnetV1/DataSet1/validation/mask_build/"

#path_learn_in="C:/Users/user/Desktop/DontTouchPLSpytorch/Notes/task2/UnetV1/DataSet1/ImgData/img/"
#path_learn_targets="C:/Users/user/Desktop/DontTouchPLSpytorch/Notes/task2/UnetV1/DataSet1/ImgData/mask_build/"
#path_vaild_in="C:/Users/user/Desktop/DontTouchPLSpytorch/Notes/task2/UnetV1/DataSet1/validation/img/"
#path_vaild_targets="C:/Users/user/Desktop/DontTouchPLSpytorch/Notes/task2/UnetV1/DataSet1/validation/mask_build/"
names_learn=os.listdir(path_learn_in)
names_valid=os.listdir(path_vaild_in)

#------------------augmentation----------------------------------------
#----------convert 3Channel colored image to 1channel class tensor--------
#----------Validation targets------------------ create a function of it augmented

list_aug_learn_in = []
list_aug_learn_targets = []

#torch.set_default_tensor_type('torch.cuda.FloatTensor')

#-----------custom dataloader below-----------------

# class DatasetCustom(data.Dataset):
#     def __init__(self,type): #def __init__(self, str): str - path or TYPE of forward
#         if((type=="train") or (type!=0)):
#             self.path_in=path_learn_in
#             self.path_targets=path_learn_targets
#             self.names=os.listdir(path_learn_in)
#         else:
#             self.path_in=path_vaild_in
#             self.path_targets=path_vaild_targets
#             self.names=os.listdir(path_vaild_in)
#
#     def __getitem__(self, item):
#         img = np.array(Image.open(path_vaild_in + names_valid[item]))
#         label = np.array(Image.open(path_vaild_targets + names_valid[item]))
#         # img = torch.from_numpy(np.asarray(img))
#         label = np.asarray(label[:, :, 0])  # [:,:,0] - first channel only because THIS SPECIFIC TASK!!
#         height = len(label[:, 0])
#         width = len(label[0, :])
#         for i in range(width):
#             for j in range(height):
#                 if (label[i, j] >= 210):
#                     label[i, j] = 1
#                 else:
#                     label[i, j] = 0
#         label = torch.from_numpy(label)
#         img = torch.from_numpy(img)
#         img = img.permute(2, 0, 1)
#
#         # return img, label.type()
#         return img.type(torch.FloatTensor), label.type(torch.FloatTensor)
#
#     def __len__(self):
#         return len(names_valid)

#----------custom dataloader above------------------



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

class DatasetCustom(data.Dataset):
    def __init__(self,type): #def __init__(self, str): str - path or TYPE of forward
        if((type=="train") or (type!=0)):
            self.path_in=path_learn_in
            self.path_targets=path_learn_targets
            self.names=os.listdir(path_learn_in)
        else:
            self.path_in=path_vaild_in
            self.path_targets=path_vaild_targets
            self.names=os.listdir(path_vaild_in)
    def __getitem__(self, item):
        img=np.array(Image.open(self.path_in+self.names[item]))
        label=np.array(Image.open(self.path_targets+self.names[item]))
        # img = torch.from_numpy(np.asarray(img))
        label=label[:,:,0]
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
        img=img.type(torch.float64)
        img=img.type(torch.FloatTensor)
        img.requires_grad=True
        # pdb.set_trace()
        # print("---IN dataloader: img.requires_grad = ", img.requires_grad)
        label=label.type(torch.int64)
        label.requires_grad=False
        return  img,label
        # print("---before return from dataloader: img.requires_grad = ", img.requires_grad)
        # return img, label # when returning from dataloader img - requires grad = false
        ### -------------
        # return img.type(torch.FloatTensor), label.type(torch.FloatTensor)
        # return img.type(torch.FloatTensor), label
    def __len__(self):
        return len(names_valid)



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
        self.classes=2 #background and other classes
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
        self.conv2d32=nn.Conv2d(in_channels=self.features*4,out_channels=self.features*4,kernel_size=3,stride=1,padding=1)
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
        self.conv2d61 = nn.Conv2d(in_channels=self.features * 16+self.features*8, out_channels=self.features * 8, kernel_size=3, stride=1, padding=1)
        self.BN6 = nn.BatchNorm2d(self.features * 8)
        self.conv2d62 = nn.Conv2d(in_channels=self.features * 8, out_channels=self.features * 8, kernel_size=3, stride=1, padding=1)

        self.TransConv2d2=nn.ConvTranspose2d(in_channels=self.features*8,out_channels=self.features*8,kernel_size=2,stride=2,padding=0)
        self.BN7=nn.BatchNorm2d(self.features*4)
        self.conv2d71 = nn.Conv2d(in_channels=self.features * 8+self.features*4, out_channels=self.features * 4, kernel_size=3, stride=1, padding=1)
        self.conv2d72 = nn.Conv2d(in_channels=self.features * 4, out_channels=self.features * 4, kernel_size=3, stride=1, padding=1)

        self.TransConv2d3=nn.ConvTranspose2d(in_channels=self.features*4,out_channels=self.features*4,kernel_size=2,stride=2,padding=0)
        self.conv2d81 = nn.Conv2d(in_channels=self.features * 4+self.features*2, out_channels=self.features * 2, kernel_size=3, stride=1, padding=1)
        self.BN8=nn.BatchNorm2d(self.features*2)
        self.conv2d82 = nn.Conv2d(in_channels=self.features * 2, out_channels=self.features * 2, kernel_size=3, stride=1, padding=1)

        self.TransConv2d4=nn.ConvTranspose2d(in_channels=self.features*2,out_channels=self.features*2,kernel_size=2,stride=2,padding=0)
        self.conv2d91 = nn.Conv2d(in_channels=self.features * 2+self.features*1, out_channels=self.features * 1, kernel_size=3, stride=1, padding=1)
        self.BN9=nn.BatchNorm2d(self.features*1)
        self.conv2d92 = nn.Conv2d(in_channels=self.features * 1, out_channels=self.features * 1, kernel_size=3, stride=1, padding=1)

        self.conv2d101=nn.Conv2d(in_channels=self.features * 1, out_channels=self.classes, kernel_size=1, stride=1, padding=0)

    # def forward(self,x):
    #     out=self.conv2dtest(x)
    #     out=self.BNtest(out)
    #     out=self.relu1(out)
    #     return out

    def forward(self,x):
        out=self.conv2d11(x)
        # pdb.set_trace()
        # print('THIS COEDE RUNS')
        out=self.BN1(out)
        # print("out.size() ",out.size())
        out=self.relu1(out)
        # print("out.size() ",out.size())
        out=self.conv2d12(out)
        # print("out.size() ",out.size())
        out=self.BN1(out)
        # print("out.size() ",out.size())
        out=self.relu1(out)
        # print("out.size() ",out.size())
        concat1=out
        out=self.maxPool1(out)
        # print("out.size() ",out.size())
        #section  #1 Above
        out=self.conv2d21(out)  #
        # print("out.size() ",out.size())
        out=self.BN2(out)
        out=self.relu1(out)
        out=self.conv2d22(out)
        out=self.BN2(out)
        out=self.relu1(out)
        concat2=out
        out=self.maxPool2(out)
        #section  #2 Above
        out=self.conv2d31(out) #
        out=self.BN3(out)
        out=self.relu1(out)
        out=self.conv2d32(out)
        out=self.BN3(out)
        out=self.relu1(out)
        concat3=out
        out=self.maxPool3(out)
        #section  #3 Above
        out=self.conv2d41(out)
        out=self.BN4(out)
        out=self.relu1(out)
        out=self.conv2d42(out)
        out=self.BN4(out)
        out=self.relu1(out)
        concat4=out
        out=self.maxPool4(out)
        #section  #4 Above
        out=self.conv2d51(out) #
        out=self.BN5(out)
        out=self.relu1(out)
        out=self.conv2d52(out)
        out=self.BN5(out)
        out=self.relu1(out)
        concat5=out
        # out=self.maxPool5(out)
        #section  #5 Above bottle neck

        out=self.TransConv2d1(out)
        out=torch.cat([out,concat4],dim=1) #probably too depends on cost function and label type represent at wich dimension should we stack
        out=self.conv2d61(out)
        out=self.BN6(out)
        out=self.relu1(out)
        out=self.conv2d62(out)
        out=self.BN6(out)
        out=self.relu1(out)
        #seciton UpSampling Associtated with #4 above

        out=self.TransConv2d2(out)
        out=torch.cat([out,concat3],dim=1) #probably too depends on cost function and label type represent at wich dimension should we stack
        out=self.conv2d71(out)
        out=self.BN7(out)
        out=self.relu1(out)
        out=self.conv2d72(out)
        out=self.BN7(out)
        out=self.relu1(out)
        #seciton UpSampling Associtated with #3 above

        # pdb.set_trace() #------------------------pdb---------
        out=self.TransConv2d3(out)
        out=torch.cat([out,concat2],dim=1) #probably too depends on cost function and label type represent at wich dimension should we stack
        out=self.conv2d81(out)
        out=self.BN8(out)
        out=self.relu1(out)
        out=self.conv2d82(out)
        out=self.BN8(out)
        out=self.relu1(out)
        #seciton UpSampling Associtated with #2 above

        out = self.TransConv2d4(out)
        out = torch.cat([out, concat1], dim=1)  # probably too depends on cost function and label type represent at wich dimension should we stack
        out = self.conv2d91(out)
        out = self.BN9(out)
        out = self.relu1(out)
        out = self.conv2d92(out)
        out = self.BN9(out)
        out = self.relu1(out)
        # seciton UpSampling Associtated with #1 above

        out=self.conv2d101(out)
        out=self.relu1(out)
        # out=torch.tensor(out)
        # pdb.set_trace()#-------------pdb----
        return out

    def load_weights(self, weights_file):
        other, ext = os.path.splitext(weights_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(weights_file, map_location=lambda storage, loc: storage))
            # stirct false\true - можно параметры загружать с определённого места
            #stirct false\true - allows us to load parameters from certain place

            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


#
# train_set=DataTrain()
# data_train=data.DataLoader(train_set,batch_size=batch_size) # train data
#
# batch_size_valid=7
# valid_data=DataValid()
# data_valid=data.DataLoader(valid_data,batch_size=batch_size_valid)
# loss=nn.CrossEntropyLoss()
# loss=nn.MSELoss()
# net=Model().to(device)

# pdb.set_trace()
# use_cuda = 0
# net=net.to(device)

# cuda.is_available()
# pdb.set_trace()
# net = net.cuda()
#plt.ion

# list_2 = []
# for param in net.named_parameters():
#     list_2.append(param)
# print(len(list_2))

#sys.exit(0)

#pdb.set_trace()


batch_size=8 #always in 2^N compatible
data_set=DatasetCustom("train")y
data_train=data.DataLoader(data_set, batch_size=batch_size)
learn_rate=5e-3
net = Model()
net=net.cuda()
loss=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(net.parameters(), lr=learn_rate)
# torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False)

print("just for check IN Y/N")
str=input()
if(str=="Y" or str=="y"):
    save_folder = "/home/std_11/MazurM/UnetV1/Parameters/"
    # save_folder="/storage/MazurM/Task1/SavedParameters/UnetV1/polygon/"
    file_saved_parameters = "Unet1v1Parameters.pth"
    torch.save(net.state_dict(), os.path.join(save_folder, file_saved_parameters))
    print("parameters SUCCESSFULLY SAVED")
    print("You saved parameters")
else:
    print("you are not save parameters")

amount_epochs=2000
# pdb.set_trace()
for ep in range(amount_epochs):
    if (not(ep % 600)):
        print("want to save parameters?")
        print("Y/N?")
        str = input()
        if (str == "Y" or str == "y"):
            save_folder = "/home/std_11/MazurM/UnetV1/Parameters/"
            # save_folder="/storage/MazurM/Task1/SavedParameters/UnetV1/polygon/"
            file_saved_parameters = "Unet1v1Parameters.pth"
            torch.save(net.state_dict(), os.path.join(save_folder, file_saved_parameters))
            print("parameters SUCCESSFULLY SAVED")
        else:
            prints("Parameters not saved")
            pass

    for i,(img,label) in enumerate(data_train):
        # img=img.requires_grad_() # not needed
        optimizer.zero_grad()
        img=img.cuda()
        label=label.cuda()
        # pdb.set_trace()
        output=net(img) # shpud be size torch.Size([BATCHSIZE, numclasses, Height,WIdht])
        error=loss(output,label)
        error.backward()
        optimizer.step()
        # a.data.cpu().numpy()[0]
        #error.data.cpu().numpy().item()
        print("epoch num ", ep)
        print(error.data.cpu().numpy().item())
        # print(error.data)
        # print("breakpoint")

print("now parameters will be saved")
save_folder="/home/std_11/MazurM/UnetV1/Parameters/"
#save_folder="/storage/MazurM/Task1/SavedParameters/UnetV1/polygon/"
file_saved_parameters="Unet1v1Parameters.pth"
torch.save(net.state_dict(), os.path.join(save_folder, file_saved_parameters))
print("check saved parameters")
amount_epochs=40
for epoch in range(amount_epochs):

    with torch.no_grad():
        amount_all_pixels=0
        true_predicted_pixels=0
        for i,(data_in, label) in enumerate(data_valid):
            data_in = data_in.type(torch.cuda.FloatTensor)
            label = label.cuda()
            label=label.unsqueeze(1)
            predict=net(data_in)
            # error=loss(predict,label)
            error=loss(predict[:,0,:,:],label)
            predict_in_batch=0
            predict_in_batch+=(label==predict[:,0,:,:]).sum()
            total_pixels_in_batch=label.size()[1]*label.size()[2]*label.size()[0]
            amount_all_pixels+=total_pixels_in_batch
            true_predicted_pixels+=predict_in_batch
        accuracy=(true_predicted_pixels.item()/amount_all_pixels)*100.0
        print("Before epoch # ", epoch)
        print("accuracy ",accuracy)
        print("------------------------------------")


    print('THIS COEDE RUNS LOL')
    for i,(data_in,label) in enumerate(data_train):
        data_in = data_in.type(torch.cuda.FloatTensor)
        label = label.cuda()
        data_in.requires_grad=True
        predict=net(data_in)
        optimizer.zero_grad()
        print(predict.size())
        #error=loss(predict,label)
        # pdb.set_trace()#----------------pdb--------------
        error=loss(predict[:,0,:,:],label)
        error.backward()
        optimizer.step()
        print("break point")
    save_folder="/home/std_11/MazurM/UnetV1/Parameters/"
	#save_folder="/storage/MazurM/Task1/SavedParameters/UnetV1/polygon/"
    file_saved_parameters="Unet1v1Parameters.pth"
    torch.save(net.state_dict(), os.path.join(save_folder, file_saved_parameters))

save_folder="/home/std_11/MazurM/UnetV1/Parameters/"
#save_folder="/storage/MazurM/Task1/SavedParameters/UnetV1/polygon/"
file_saved_parameters="Unet1v1Parameters.pth"
torch.save(net.state_dict(), os.path.join(save_folder, file_saved_parameters))
#sake of backup
