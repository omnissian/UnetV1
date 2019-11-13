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
path_learn_in="C:/Users/user/Desktop/DontTouchPLSpytorch/nets/unet1v1/data_example/in/"
path_learn_label="C:/Users/user/Desktop/DontTouchPLSpytorch/nets/unet1v1/data_example/label/"
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
    def __init__(self):
        super(Model,self).__init__()
        self.conv2d11=nn.Conv2d(in_channels=3,out_channels=64, kernel_size=3, stride=1,padding=3)
        self.relu1=nn.ReLU()
        self.conv2d12=nn.Conv2d(in_channels=64,out_channels=64, kernel_size=3, stride=1,padding=3) #>>>>
        self.maxPool1=nn.MaxPool2d(kernel_size=2,stride=1) #--------drop ----maxpool---

        self.conv2d21=nn.Conv2d(in_channels=64,out_channels=128, kernel_size=3, stride=1,padding=3)
        self.conv2d22 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=3)#>>>
        self.maxPool2=nn.MaxPool2d(kernel_size=2,stride=1) #--------drop ----maxpool---

        self.conv2d31=nn.Conv2d(in_channels=128,out_channels=256, kernel_size=3, stride=1,padding=3)
        self.conv2d32=nn.Conv2d(in_channels=256,out_channels=256, kernel_size=3, stride=1,padding=3)#>>
        #after it save out for concat1
        self.maxPool3=nn.MaxPool2d(kernel_size=2,stride=1)#--------drop ----maxpool---

        self.conv2d41=nn.Conv2d(in_channels=256,out_channels=512, kernel_size=3,stride=1,padding=3)
        self.conv2d42=nn.Conv2d(in_channels=512,out_channels=512, kernel_size=3,stride=1,padding=3)#--------concat it (4)

        self.maxPool4=nn.MaxPool2d(kernel_size=2, stride=1)#--------drop ----maxpool---

        self.conv2d51=nn.Conv2d(in_channels=512,out_channels=1024, kernel_size=3,stride=1,padding=3)
        self.conv2d52=nn.Conv2d(in_channels=1024,out_channels=1024, kernel_size=3,stride=1,padding=3)
#----------------------------------------------------------------
        # self.ConvTrans2d1=nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=3) #<
        self.ConvTrans2d1=nn.ConvTranspose2d(in_channels=1024,out_channels=1024,kernel_size=3,stride=1,padding=0) #< #--------concat it (4)
        # --------concat it (4)
        # #self.concat1   --will be write in forward section
        self.conv2dr51=nn.Conv2d(in_channels=768,out_channels=256, kernel_size=3,stride=1,padding=3)
        self.conv2dr52=nn.Conv2d(in_channels=256,out_channels=256, kernel_size=3,stride=1,padding=3)
        # self.maxPool1=nn.MaxPool2d(kerne_size=2,stride=1) #--------drop ----maxpool---

        # self.ConvTrans2d2=nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3,stride=1,padding=3) # <<
        self.ConvTrans2d2=nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3,stride=1,padding=0) # <<
        # #self.concat2   --will be write in forward section
        self.conv2d61=nn.Conv2d(in_channels=384,out_channels=128, kernel_size=3,stride=1,padding=3)
        self.conv2d62=nn.Conv2d(in_channels=128,out_channels=128, kernel_size=3,stride=1,padding=3)

        # self.ConvTrans2d3=nn.ConvTranspose2d(in_channels=192, out_channels=64, kernel_size=3,stride=1,padding=3) # <<
        self.ConvTrans2d3=nn.ConvTranspose2d(in_channels=192, out_channels=64, kernel_size=3,stride=1,padding=0) # <<
        # #self.concat3   --will be write in forward section
        self.conv2d61=nn.Conv2d(in_channels=192,out_channels=64, kernel_size=3,stride=1,padding=3)
        self.conv2d61=nn.Conv2d(in_channels=64,out_channels=64, kernel_size=3,stride=1,padding=3)

        self.conv2d61=nn.Conv2d(in_channels=64,out_channels=3, kernel_size=3,stride=1,padding=3)
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
        concat1=out # ---------------------------------->
        out=self.maxPool1(out)
        print("out=self.maxPool1(out)")
        print(out.size())
        out=self.conv2d21(out)
        print("out=self.conv2d21(out)")
        print(out.size())
        out=self.conv2d22(out)
        print("out=self.conv2d22(out)")
        print(out.size())
        concat2=out  # ---------------------------------->>
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
        concat3=out # ---------------------------------->>>
        print(concat3.size()) #---------this we will concat 3
        out=self.maxPool3(out)
        print("out=self.maxPool3(out)")
        print(out.size())
        out=self.conv2d41(out)
        print("out=self.conv2d41(out)")
        print(out.size())
        out=self.conv2d42(out)
        print("out=self.conv2d42(out)")
        print(out.size())
        #------------------------------
        print("--------------concatenate this now!-----------")
        out=self.ConvTrans2d1(out) #---------this we will concatenate with concat 3
        print("out=self.ConvTrans2d1(out)")
        print(out.size())
        print("----------------------------------------------")

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















#
#
#
#
#
# # #--------------------###############################################--------------------------------------------------------------------------
# # Absolut Halal
# # https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_feedforward_neuralnetwork/
# #----------------------
# # import torch
# # import torch.nn as nn
# # import torchvision.transforms as transforms
# # import torchvision.datasets as dsets
# # import matplotlib.pyplot as plt
# # import random
# # import os
#
#
# train_dataset=dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(),download=True)
# test_dataset=dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
# # print(train_dataset.test_data.size()) # old name - data
# print(train_dataset.data.size())  # old name - targets
# print(train_dataset.train_labels.size())
# print(test_dataset.data.size())
# batch_size=9 #256
# epochs=400
# train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
# test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)
#
# print(type(train_loader)) #---------------------check shape
# print("hello")
#
# class Model(nn.Module):
#     def __init__(self):
#         super(Model,self).__init__()
#         self.conv2d1=nn.Conv2d(in_channels=1,out_channels=20,kernel_size=4,stride=1,padding=3)
#         self.relu1=nn.ReLU()
#         self.maxPool1=nn.MaxPool2d(kernel_size=2,stride=1)
#         self.conv2d2=nn.Conv2d(in_channels=20,out_channels=10,kernel_size=4,stride=1,padding=3)
#         self.relu2=nn.ReLU()
#         self.maxPool2=nn.MaxPool2d(kernel_size=2,stride=1)
#         self.fc1=nn.Linear(10*32*32,10)
#     def forward(self,x):
#         out=self.conv2d1(x)
#         out=self.relu1(out)
#         out=self.maxPool1(out)
#         out=self.conv2d2(out)
#         out=self.relu2(out)
#         out=self.maxPool2(out)
#         # print("print(out.size(0)) ", out.size(0))
#         out=out.view(out.size(0),-1)
#         out=self.fc1(out)
#         return out
#     # base_file="C:\Users\user\Desktop\DontTouchPLSpytorch\Notes\SavedParameters\Cnn1\Cnn1V1Parameters"
#     def load_weights(self, weights_file):
#         other, ext = os.path.splitext(weights_file)
#         if ext == '.pkl' or '.pth':
#             print('Loading weights into state dict...')
#             self.load_state_dict(torch.load(weights_file, map_location=lambda storage, loc: storage))
#             #stirct false\true - можно параметры загружать с определённого места
#             print('Finished!')
#         else:
#             print('Sorry only .pth and .pkl files supported.')
#
#
#
#
# myModel=Model()
# cost_func=nn.CrossEntropyLoss()
# learn_rate=0.0006
# optimizer=torch.optim.SGD(myModel.parameters(), lr=learn_rate)
# #---------------------------------------------------------------------
# # path='C:\\Users\\Anatoly\\Desktop\\Work1\\plgn\\data10by10digitstest\\forlearn7\\'
# test_loader_manual=torch.utils.data.DataLoader(dataset=test_dataset,shuffle=False)
# # save_folder="C:\Users\user\Desktop\DontTouchPLSpytorch\Notes\SavedParameters\Cnn1"
#
# save_folder="C:\\Users\\user\\Desktop\\DontTouchPLSpytorch\\Notes\\SavedParameters\\Cnn1\\"
#
# file="Cnn1V1Parameters.pth"
#
# plt.ion # disable interactive mod
# #--------------creating new data set for **----------------------
#
# print("here we go")
#
#
# #-----------------try to pack a tensor----START---------
#
#
#
#
#
#
#
# #-----------------try to pack a tensor----END---------
#
#
# img_pack=[]
# iter=0
# for i, (image,target) in enumerate(test_loader_manual):
#     img_pack.append((image, target))
#     iter+=1
#
# print("do you want to train the neural net?")
# print("'1' for YES OR  '0' for NO")
# answer=int(input())
# #--------------creating new data set for **----------------------
# if(answer):
#     for n_epoch in range (epochs):
#         if((n_epoch%1)==0): # change the denominator
#             # total=len(test_loader)
#             # print("print(len(test_loader)) ", len(test_loader))
#             predicted=0
#             total=0
#             with torch.no_grad():
#                 for c, (input_test,targets) in enumerate (test_loader):
#                     size_labels = len(targets)
#                     # print("print(targets.size(0))", targets.size(0))  # 20 workable original
#                     # print("print(targets.size())", targets.size()[0])  # 20 - workable
#                     # # print("print(targets.size())", targets.size().item())  # 20 -doesnt work
#                     output = myModel(input_test)
#                     _, predict=torch.max(output,1)
#                     # predicted=(predict==targets).sum()
#                     for x in range(size_labels):
#                         total+=1
#                         # print("print(targets.data[x].item())= ", targets.data[x].item())
#                         if(targets.data[x].item()==predict.data[x].item()):
#                             predicted+=1
#                 accuracy=(predicted/total)*100
#                 print("accuracy= ",accuracy, " after epoch ", n_epoch)
#         print("train started")
#         for i,(input_train,targets) in enumerate(train_loader):
#             # if(i%100):
#             #     print("iteration: ",i)
#             input_train=input_train.requires_grad_()
#             optimizer.zero_grad()
#             predict=myModel(input_train)
#             loss=cost_func(predict,targets)
#             # print("loss.item() ", loss.item())
#             loss.backward()
#             optimizer.step()
#         print("training at epoch№",n_epoch, " ended")
#     print("The Neural Net is Trained")
# else:pass
# # -----------saving parameters-----------
#
# # torch.save(myModel.state_dict(), os.path.join(save_folder, file))
# # print("LEARNED* PARAMETERS SAVED!!!! CONGRATS!!!!")
# # print("COPY THEM!!! DONT LOST THEM!!!")
# # print("want to save parametes? Y/N")
# print("want to save parametes? '1' for YES OR  '0' for NO")
# # answer=str(input())
# answer=int(input())
# if(answer):
#     torch.save(myModel.state_dict(), os.path.join(save_folder, file))
# else:
#     print("do you want to load the weights of net?")
#     print("'1' for YES OR  '0' for NO")
#     answer = int(input())
#     if(answer):
#         myModel.load_weights(save_folder + file)
#     else:pass
#
#
# # print(test_loader_manual.len())
# # print(test_dataset.__len__())
# size_test=test_dataset.__len__()
# # fig=plt.figure()
#
# num=0
# while(0<=num<size_test):
#     fig = plt.figure()
#     # plt.clf()
#     print("Enter the number of image that you want to test")
#     print(" in range 0...",size_test)
#     num=int(input())
#     output=myModel(img_pack[num][0])
#     _,predict=torch.max(output,1)
#     print("*******************************************")
#     print("********predicted Number = ", predict,"****")
#     print("*******************************************")
#     fig.canvas.set_window_title(str(img_pack[num][1].item()))
#     # fig=plt.imshow(img_pack[num][0][0].numpy()[0],cmap='gray')
#     wtf=img_pack[num][0][0].numpy()[0]
#     plt.imshow(img_pack[num][0][0].numpy()[0],cmap='gray')
#     # fig.draw()
#     plt.show()
#     # fig.canvas.draw_idle()
#     # plt.pause(1)
#     # plt.show()
#     # plt.close()
#
# i=0
# fig=plt.figure()
# output=myModel(img_pack[0][0])
# _, predict = torch.max(output, 1)
# print(predict.item())
# fig=plt.figure(0)
# fig.canvas.set_window_title(str(img_pack[i][1].item()))
# wtf=img_pack[i][0][0]
# print("wtf.size() = ",wtf.size())
# fig=plt.imshow(img_pack[i][0][0].numpy()[0],cmap='gray')
#
#
# plt.show()
#
#
# # save_folder="C:\Users\user\Desktop\DontTouchPLSpytorch\Notes\SavedParameters\Cnn1"
# # file="Cnn1V1Parameters"
# # if(answer):
# #     torch.save(myModel.state_dict(), os.path.join(save_folder, file))
#
#
# #-----------saving parameters-----------
# print("enter negative value for exit")
# c=0
# test_loader_manual=torch.utils.data.DataLoader(dataset=test_dataset)
#
# # print(myModel.parameters())
# # print("parameters in first 0 tensor of parameters ", list(myModel.parameters())[0].size())
# # print("------------------------------------")
# # print("Ere we go 0! : list(myModel.parameters())[0] ", list(myModel.parameters())[0])
# # print("------------------------------------")
# # print("parameters in second 1 tensor of parameters ", list(myModel.parameters())[1].size())
# # print("Ere we go 1! : list(myModel.parameters())[1] ", list(myModel.parameters())[1])
#
# print("end")
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
