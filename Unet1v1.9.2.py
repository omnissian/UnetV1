


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
    def __getitem__(self, index):
        # img = cv2.imread(self.img_paths[index])
        img=Image.open(path_valid_in+names_valid_in[index])
        label=Image.open(path_valid_label+names_valid_in[index])
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
    imgs = []
    targets = []
    for sample in batch:
        tmpr_l=sample[0]
        tmpr_l=tmpr_l.permute(2,0,1)
        print("type(tmpr_l ",type(tmpr_l))
        # tmpr_l=torch.FloatTensor(tmpr_l) # added 18/11/2019

        tmpr_r=sample[1]
        tmpr_r=tmpr_r.permute(2,0,1)
        # tmpr_r=torch.FloatTensor(tmpr_r) # added 18/11/2019

        imgs.append(tmpr_l)
        # targets.append((tmpr_r))
        targets.append(tmpr_r)
        # imgs.append(torch.FloatTensor(sample[0]).permute(2, 0, 1))
        # targets.append(torch.FloatTensor([[sample[1]]]))
        # targets.append(torch.FloatTensor(sample[1].permute(2,0,1)))
        print("type(imgs) " , type(imgs))
        print("type(targets) ", type(targets))


    # return torch.stack(imgs, 0), targets #
    return torch.stack(imgs,0), torch.stack(targets, 1)



train_dataset=MyDataset()
data_train_loader=data.DataLoader(train_dataset, batch_size,num_workers=num_workers,shuffle=True,collate_fn=detection_collate,pin_memory=True,drop_last=True)
valid_dataset=ValidationData()
data_valid_loader=data.DataLoader(valid_dataset, batch_size, num_workers=num_workers,shuffle=False, collate_fn=detection_collate, pin_memory=True, drop_last=True)


# data_example_train=iter(data_train_loader)

# print(data_example_train.next())
print("wtf")
#------------------need to add batch norm, need to add activation layers

class Model(nn.Module):
    img_channels=3
    # features=64
    def __init__(self):
        super(Model,self).__init__()
        # Model.to('cuda')
        img_channels=3
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

        self.conv2d101=nn.Conv2d(in_channels=features*1, out_channels=img_channels,kernel_size=1,stride=1,padding=0) #last conv layer



        #no activation after this layer
    def forward(self,x):
        print("type(x) ", type(x))
        x
        # x.to(device)
        print(x.size())
        out=self.conv2d11(x)
        print("out=self.conv2d11(x)")
        print(out.size())
        out=self.relu1(out) #<< RELU
        out=self.conv2d12(out)
        print("out=self.conv2d12(out)")
        print(out.size())
        out=self.relu1(out) #<< RELU
        concat1=out # ----------------------------------1>>>> MAX POOL 1
        print("size concat1", concat1.size())
        out=self.maxPool1(out)
        print("out=self.maxPool1(out)")
        print(out.size())
        out=self.conv2d21(out)
        print("out=self.conv2d21(out)")
        print(out.size())
        out=self.relu1(out) #<< RELU
        out=self.conv2d22(out)
        print("out=self.conv2d22(out)")
        print(out.size())
        concat2=out  # ---------------------------------->>> MAX POOL 2
        print("size concat2", concat2.size())
        out=self.relu1(out) #<< RELU
        out=self.maxPool2(out)
        print("out=self.maxPool2(out)")
        print(out.size())
        out=self.conv2d31(out)
        out=self.relu1(out) #<< RELU
        print("out=self.conv2d31(out)")
        print(out.size())
        out=self.conv2d32(out)
        out=self.relu1(out) #<< RELU
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
        out=self.relu1(out) #<< RELU
        print("out=self.conv2d41(out)")
        print(out.size())
        out=self.conv2d42(out)
        out=self.relu1(out) #<< RELU
        print("out=self.conv2d42(out)")
        print(out.size())
        concat4=out # ---------------------------------->> MAX POOL 4 (512+1024 = 1536)
        out=self.maxPool4(out)    # max pool 4
        out=self.conv2d51(out)
        out=self.relu1(out) #<< RELU
        print("out=self.conv2d51(out)")
        print(out.size())
        out=self.conv2d52(out)
        out=self.relu1(out) #<< RELU
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
        out=self.relu1(out) #<< RELU
        print("out=self.conv2d61(out)")
        print(out.size())
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


        print("size of out ",out.size())
        print("here we go")
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
# wtf1=enumerate(data_train_loader)
# wtf2=iter(data_train_loader).next()
#
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

#-------------found number of all classes-----------------
original_pixels=[]
adds=0
#---test!!! adds=1
adds=1


#---test!!! adds=1
for iter_valid_data, (input_valid, targets_valid) in enumerate(data_valid_loader):
    original_pixels=(targets_valid.permute(1,0,2,3)).data[0][:,0,0]

    # ---test!!! adds=1
    test_cat=original_pixels
    # test_cat=torch.tensor([0,0,0],device='cpu')
    print("test_cat.size() ",test_cat.size())
    original_pixels=torch.stack([original_pixels,original_pixels],dim=0)
    print("original_pixels.size() ",original_pixels.size())
    # original_pixels=torch.cat([original_pixels.data[0],test_cat],dim=0)
    # torch.cat([original_pixels,original_pixels],dim=0]
    print("original_pixels.size() ",original_pixels.size())
    print("len(original_pixels) ",len(original_pixels))
    # ---test!!! adds=1
    tmp = (targets_valid.permute(1, 0, 2, 3)).data[0]
    channels = targets_valid.permute(1, 0, 2, 3).size()
    width = channels[3]
    height = channels[2]
    channels = channels[1]
    # wtf1=targets_valid.permute(1,0,2,3)
    print("targets_valid.size() ",targets_valid.size())
    for i in range(batch_size):
        # original_pix=np.zeros((width,height,channels))
        # original_pixels=torch.tensor([0,0,0]) # for rgb
        for i in range(width):
            for j in range(height):
                pix=tmp.data[:,i,j]
                WTF_len=len(original_pixels)
                if(adds):
                    is_in=False
                    for ip in range (len(original_pixels)):
                        # dict_cur_pix=original_pixels.data[0]
                        left=original_pixels.data[ip]
                        right=pix.data
                        print(right)
                        print(left)
                        wtf_logic = (torch.all(torch.eq(pix, original_pixels.data[ip]))).item()
                        if(wtf_logic):
                            is_in=True
                            break
                        # else:
                        #     right=right.unsqueeze(0)
                        #     print("left= ",left)
                        #     print("right= ",right)
                        #
                        #     # original_pixels=torch.cat([original_pixels,pix],dim=0)
                        #     original_pixels=torch.cat([original_pixels,right],dim=0)
                        #     break
                    if(not is_in):
                        new_pix=pix.unsqueeze(0)
                        original_pixels=torch.cat([original_pixels,new_pix], dim=0)
                        pass

                else:
                    wtf_logic=(torch.all(torch.eq(pix,original_pixels))).item()
                    if(wtf_logic):
                        pass
                    else:
                        torch.stack([original_pixels,pix],dim=0)
                        adds+=1;

                    # wtf_logic=(torch.all(torch.eq(pix,original_pixels))).data[0]

                # print("original_pixels.size() ",original_pixels.size())
                # print("original_pixels[0].size() ",original_pixels[0].size())
                # torch.eq(pix,)
    print("stop")
            # pixel_ch = 0
            # img_out=tmp.data[ch]
            # width,height =img_out.size()
            # for i in range(width):
            #     for j in range (height):
            #         img_out
        #----------------------------------------------------
        # for ch in range(channels):
        #     pixel_ch = 0
        #     img_out=tmp.data[ch]
        #     width,height =img_out.size()
        #     for i in range(width):
        #         for j in range (height):
        #             img_out

# -------------found number of all classes-----------------


prin("check all stuff")
if (answer):
    print("please in the number>0 of train epochs")
    # epochs=int(input())
    epochs=9
    for num_epoch in range(epochs):
        if(epochs>0):
            predicted=0
            total=0
            with torch.no_grad():
                for iter_valid_data,(input_valid, targets_valid) in enumerate(data_valid_loader):
                    size_labels=len(targets_valid)
                    input_valid=input_valid.to(device) # cuda() #----------to CUDA--------GPU---------
                    input_valid=input_valid.to(device=device, dtype=torch.float)
                    # input_valid=input_valid.float()
                    # targets_valid=targets_valid.to(device)
                    targets_valid=targets_valid.to(device=device, dtype=torch.float)
                    output=net(input_valid) # original -----here some problem---------------
                    print("type(output) ",type(output))
                    print("output.size() ",output.size())
                    # output=output.to(device=device, dtype=torch.float)
                    # _,net_predict=torch.max(output,1)
                    # _,net_predict=torch.max(output,0)
                    # output=output.permute(1,0,2,3)
                    targets_valid=targets_valid.permute(1,0,2,3)
                    print("output.size() ", output.size())
                    print("output.data[1].size() ", output.data[1].size())
                    print("size_labels= ", size_labels)
                    for x in range(size_labels):
                        total+=1
                        if(torch.equal(output.data[x],targets_valid.data[x])):
                            predicted+=1
                    # torch.eq(targets_valid,)

                    # torch.eq(a, b)
                    # for x in range( size_labels):
                    #     if(output.data[x].item()==targets_valid.data[x].item()):
                    #         predicted+=1
                    # net_predict=net_predict.to(device=device, dtype=torch.float)
                    # print("net_predict.size() ",net_predict.size())
                    # net_predict=net_predict.to('cuda')
                    #predicted=(predict==targets).sum()
                    #---------------------------------
                    # for x in range(size_labels):
                    #     print("type(targets_valid) ", type(targets_valid)) # [3, 4, 256, 256]  targets_valid
                    #     print("targets_valid.size() ",targets_valid.size())
                    #     print("type(net_predict) ", type(net_predict)) #  [4, 256, 256]   net_predict
                    #     print("net_predict.size() ", net_predict.size())
                    #     total+=1
                    #     if(targets_valid.data[x].item()==net_predict.data[x].item()):
                    #         predicted+=1
                    accuracy=(predicted/total)*100.0
                    print("After epoch ", num_epoch,"Accuracy= ", accuracy)
        print("---->train epoch started<------")

        for i, (input_train, targets) in enumerate(data_train_loader):
            print(targets[1].data.getitem())
            input_train=input_train.to(device=device, dtype=torch.float)
            targets=targets.to(device=device, dtype=torch.float)
            input_train = input_train.requires_grad_()
            # input_train.to('cuda') #----------to GPU-------------GPU-------------
            # targets=targets.to('cuda')
            optimizer.zero_grad()
            predict = net(input_train)
            print("predict.size() ",predict.size())
            print("targets.size()= ",targets.size())
            targets=targets.permute(1,0,2,3)

            loss = cost_func(predict, targets)
            loss.backward()
            optimizer.step()
            # input_train.float()
            # net(input)
            # print(i)
            # print("one batch finished")
        print("training Epoch ?",num_epoch," ended")
    print("All training epochs are passed")
else:pass # not to train

# for i, (input_train, targets) in enumerate(data_train_loader):
#     input_train=input_train.requires_grad_()
#     optimizer.zero_grad()
#     predict=net(input_train)
#     loss=cost_func(predict,targets)
#     loss.backward()
#     optimizer.step()
#     # input_train.float()
#     # net(input)
#     # print(i)
#     print("one batch finished")






print("go1")
net(wtf1.next())
# net.forward()
print("net",net)
# net(data_example_train.next())


print("hello")
