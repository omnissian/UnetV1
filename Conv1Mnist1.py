# import os
# # C:\Users\user\Desktop\НПЗ_мосты\Texas\union1\test_datum\4957\db
# # path='/storage61/Platform/Melnichenko/db/lidar_only_spz/Mazur/Texas/union1/test_datum/4957/db'
# path='C:/Users/user/Desktop/НПЗ_мосты/Texas/union1/test_datum/4957/db'
#
# list_of_names=os.listdir(path)
# for i in range(list_of_names.__len__()):
# 	print(list_of_names[i])
# 	input()












# ##-------------------------------------------##############################################-----------------------------------

#------------------------------------------------------------------
#
# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# # C:\Users\user\Desktop\НПЗ_мосты\NewJersey\2007FEMA\rezult\db\8_512_1536_textur.tiff
# # im = Image.open("/storage/MazurM/Task1/ImgData/img/205_0_0.tiff")
# im = Image.open("C:/Users/user/Desktop/НПЗ_мосты/NewJersey/2007FEMA/rezult/db/8_512_1536_textur.tiff")
#
# # plt.imshow(
# print(type(im))
# x=np.array(im)
# print(x)
# print(type(x))
# print(x.shape)
# plt.imshow(x)
# plt.show()
# print("wtf")
# ##-------------------------------------------##############################################------------------------------------
#
# ##----------------------------------------------------------------------------------------------
# #
# # import os
# # import codecs
# # import numpy as np
# # import re
# #
# # # 1 arg - path to files that should be renamed
# # # 2 arg - path to file that contains a name in every line\string
# # # 3 arg - file name
# # #path='C:\\Users\\user\\Desktop\\DontTouchPLSpytorch\\Notes\\Scripts\\PyRename\\rezult_Indiana_Whitting\\',
# # # def parser (path='C:/Users/user/Desktop/DontTouchPLSpytorch/Notes/Scripts/PyRename/rezult_Indiana_Whitting/',
# # #             path2='C:\\Users\\user\\Desktop\\DontTouchPLSpytorch\\Notes\\Scripts\\PyRename\\',
# # #             filename="all_spz_las_file.txt"):
# # # def parser (path='C:/Users/user/Desktop/DontTouchPLSpytorch/Notes/task/California/CaliforniaTestDatum/4893/result/img/',
# # #             path2='C:/Users/user/Desktop/DontTouchPLSpytorch/Notes/task/California/CaliforniaTestDatum/4893/',
# # #             filename="all_spz_las_file.txt"):
# # def parser (path='C:/Users/user/Desktop/DontTouchPLSpytorch/Notes/task/California/LosAngeles/test_datum/4269/rezult/images/',
# #             path2='C:/Users/user/Desktop/DontTouchPLSpytorch/Notes/task/California/LosAngeles/test_datum/4269/rezult/',
# #             filename="all_spz_las_file.txt"):
# #     # data_set = []
# #     file_names=codecs.open((path2+filename),'r')
# #     list_of_new_names=file_names.readlines()
# #     # len=list_of_new_names.__len__()
# #
# #     list_of_old_names=os.listdir(path) # deletable
# #     len_old=list_of_old_names.__len__()
# #     iterator=0
# #     # using old file names as indices for iterator at number of line in file contains new names as a lines in it
# #     for i in range(list_of_old_names.__len__()):
# #         list_of_old_names[i]=int(list_of_old_names[i].split(".")[0])
# #
# #     file_names_current=os.listdir(path)
# #     print(path+list_of_new_names[0])
# #     for i in range(len_old):
# #         k=i+1 # new_name=new_name.replace('\n','')
# #         # new_name=list_of_new_names[i].replace('\n','')
# #         wtf=list_of_new_names[(list_of_old_names[i]-1)]
# #         new_name=list_of_new_names[(list_of_old_names[i]-1)].replace('\n','')+'.png'
# #         # old_name_f=(path+str(list_of_old_names[i])+'.png')
# #         old_name=str(list_of_old_names[i])+'.png'
# #         os.rename((path+old_name),(path+new_name))
# #         #-----------------------------
# #         # old_name_f=(path+str(k)+'.png')
# #         # new_name_f=(path+new_name+'.png')
# #         # g=0
# #         # os.rename((path+str(k)+'.png'),(path+new_name+'.png'))
# #     #
# #     # for i in range(file_names_current.__len__()):
# #     #     file_names_current[i]=file_names_current[i].split(".")[0]
# #     #     print("file_names_current[i]",file_names_current[i])
# #     #     print("type(file_names_current[i])",type(file_names_current[i]))
# #     #     print("str(1)=",str(1))
# #     #     print("int(file_names_current[i]) ",int(file_names_current[i]))
# #     #     current=int(file_names_current[i])
# #     #     if(i==int(file_names_current[i])):
# #     #         print(file_names_current[i]," == ", i+1)
# #     #
# #     # for file_name_current in os.listdir(path):
# #     #     wtf=(path+file_name_current)
# #     #     new_name=(list_of_new_names[iterator]+'.png')
# #     #     new_name=new_name.replace('\n','')
# #     #     new_name=path+new_name
# #     #     os.rename(wtf,new_name)
# #     #     iterator+=1
# #     # for filename in os.listdir(path):
# #     #     tmp_path = path + filename
# #     #     file_obj = codecs.open(tmp_path, 'r')
# #     #     list_obj = file_obj.readlines()
# #     #     example = np.array([])
# #     #     tmp_splitted = []
# #     #     # print(list_obj)
# #     #     for string_n in list_obj:
# #     #         tmp = re.split('[^0-1]', string_n)
# #     #         tmp_splitted.append(tmp[0])
# #     #     for line_n in tmp_splitted:
# #     #         for char_n in line_n:
# #     #             if ((int(char_n))):
# #     #                 # example.append(1)
# #     #                 example=np.append(example,1)
# #     #             else:
# #     #                 # example.append(0)
# #     #                 example=np.append(example,0)
# #     #     data_set.append(example)
# #     #     file_obj.close()
# #     # print(list_of_names[0])
# #
# # #----------------------------------------------------------------------------------------
# # # for put img names to txt file: every img name is a new line in text file
# # # 1 First argument - path to folder with images
# # # 2 Second argument - path to folder with txt file
# # # 3 third argument - name of txt file
# #
# #
# # def writer1 (path='C:/Users/user/Desktop/DontTouchPLSpytorch/Notes/task/California/LosAngeles/test_datum/4269/rezult/images/',
# #             path2='C:/Users/user/Desktop/DontTouchPLSpytorch/Notes/task/California/LosAngeles/test_datum/4269/rezult/',
# #             filename="hehe.txt"):
# #     # data_set = []
# #     # file_names=codecs.open((path2+filename),'r')
# #     # list_of_new_names=file_names.readlines()
# #     list_of_img_names=os.listdir(path)
# #     with open((path2+filename), 'w') as file:
# #         for i in range(list_of_img_names.__len__()):
# #             name=list_of_img_names[i].split('.')[0]
# #             file.write(name)
# #             file.write('\n')
# #
# # writer1()
# # print("finished")
#
#
# #--------------------###############################################--------------------------------------------------------------------------
# Absolut Halal
# https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_feedforward_neuralnetwork/
#----------------------
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
import random
import os


train_dataset=dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(),download=True)
test_dataset=dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
# print(train_dataset.test_data.size()) # old name - data
print(train_dataset.data.size())  # old name - targets
print(train_dataset.train_labels.size())
print(test_dataset.data.size())
batch_size=9 #256
epochs=400
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

print(type(train_loader)) #---------------------check shape
print("hello")

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv2d1=nn.Conv2d(in_channels=1,out_channels=20,kernel_size=4,stride=1,padding=3)
        self.relu1=nn.ReLU()
        self.maxPool1=nn.MaxPool2d(kernel_size=2,stride=1)
        self.conv2d2=nn.Conv2d(in_channels=20,out_channels=10,kernel_size=4,stride=1,padding=3)
        self.relu2=nn.ReLU()
        self.maxPool2=nn.MaxPool2d(kernel_size=2,stride=1)
        self.fc1=nn.Linear(10*32*32,10)
    def forward(self,x):
        out=self.conv2d1(x)
        out=self.relu1(out)
        out=self.maxPool1(out)
        out=self.conv2d2(out)
        out=self.relu2(out)
        out=self.maxPool2(out)
        # print("print(out.size(0)) ", out.size(0))
        out=out.view(out.size(0),-1)
        out=self.fc1(out)
        return out
    # base_file="C:\Users\user\Desktop\DontTouchPLSpytorch\Notes\SavedParameters\Cnn1\Cnn1V1Parameters"
    def load_weights(self, weights_file):
        other, ext = os.path.splitext(weights_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(weights_file, map_location=lambda storage, loc: storage))
            #stirct false\true - можно параметры загружать с определённого места
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')




myModel=Model()
cost_func=nn.CrossEntropyLoss()
learn_rate=0.0006
optimizer=torch.optim.SGD(myModel.parameters(), lr=learn_rate)
#---------------------------------------------------------------------
# path='C:\\Users\\Anatoly\\Desktop\\Work1\\plgn\\data10by10digitstest\\forlearn7\\'
test_loader_manual=torch.utils.data.DataLoader(dataset=test_dataset,shuffle=False)
# save_folder="C:\Users\user\Desktop\DontTouchPLSpytorch\Notes\SavedParameters\Cnn1"

save_folder="C:\\Users\\user\\Desktop\\DontTouchPLSpytorch\\Notes\\SavedParameters\\Cnn1\\"

file="Cnn1V1Parameters.pth"

plt.ion # disable interactive mod
#--------------creating new data set for **----------------------

# image,target=iter(test_loader_manual).next()  #---------------------check shape for test
image,target=iter(train_loader).next()  #---------------------check shape for train
print(type(image))
print("image.size() = ",image.size())
print(target[0].item())
img_pack=[]
iter=0
for i, (image,target) in enumerate(test_loader_manual):
    img_pack.append((image, target))
    iter+=1

print("do you want to train the neural net?")
print("'1' for YES OR  '0' for NO")
answer=int(input())
#--------------creating new data set for **----------------------
if(answer):
    for n_epoch in range (epochs):
        if((n_epoch%1)==0): # change the denominator
            # total=len(test_loader)
            # print("print(len(test_loader)) ", len(test_loader))
            predicted=0
            total=0
            with torch.no_grad():
                for c, (input_test,targets) in enumerate (test_loader):
                    size_labels = len(targets)
                    # print("print(targets.size(0))", targets.size(0))  # 20 workable original
                    # print("print(targets.size())", targets.size()[0])  # 20 - workable
                    # # print("print(targets.size())", targets.size().item())  # 20 -doesnt work
                    output = myModel(input_test)
                    _, predict=torch.max(output,1)
                    # predicted=(predict==targets).sum()
                    for x in range(size_labels):
                        total+=1
                        # print("print(targets.data[x].item())= ", targets.data[x].item())
                        if(targets.data[x].item()==predict.data[x].item()):
                            predicted+=1
                accuracy=(predicted/total)*100
                print("accuracy= ",accuracy, " after epoch ", n_epoch)
        print("train started")
        for i,(input_train,targets) in enumerate(train_loader):
            # if(i%100):
            #     print("iteration: ",i)
            input_train=input_train.requires_grad_()
            optimizer.zero_grad()
            predict=myModel(input_train)
            loss=cost_func(predict,targets)
            # print("loss.item() ", loss.item())
            loss.backward()
            optimizer.step()
        print("training at epoch№",n_epoch, " ended")
    print("The Neural Net is Trained")
else:pass
# -----------saving parameters-----------

# torch.save(myModel.state_dict(), os.path.join(save_folder, file))
# print("LEARNED* PARAMETERS SAVED!!!! CONGRATS!!!!")
# print("COPY THEM!!! DONT LOST THEM!!!")
# print("want to save parametes? Y/N")
print("want to save parametes? '1' for YES OR  '0' for NO")
# answer=str(input())
answer=int(input())
if(answer):
    torch.save(myModel.state_dict(), os.path.join(save_folder, file))
else:
    print("do you want to load the weights of net?")
    print("'1' for YES OR  '0' for NO")
    answer = int(input())
    if(answer):
        myModel.load_weights(save_folder + file)
    else:pass


# print(test_loader_manual.len())
# print(test_dataset.__len__())
size_test=test_dataset.__len__()
# fig=plt.figure()

num=0
while(0<=num<size_test):
    fig = plt.figure()
    # plt.clf()
    print("Enter the number of image that you want to test")
    print(" in range 0...",size_test)
    num=int(input())
    output=myModel(img_pack[num][0])
    _,predict=torch.max(output,1)
    print("*******************************************")
    print("********predicted Number = ", predict,"****")
    print("*******************************************")
    fig.canvas.set_window_title(str(img_pack[num][1].item()))
    # fig=plt.imshow(img_pack[num][0][0].numpy()[0],cmap='gray')
    wtf=img_pack[num][0][0].numpy()[0]
    plt.imshow(img_pack[num][0][0].numpy()[0],cmap='gray')
    # fig.draw()
    plt.show()
    # fig.canvas.draw_idle()
    # plt.pause(1)
    # plt.show()
    # plt.close()

i=0
fig=plt.figure()
output=myModel(img_pack[0][0])
_, predict = torch.max(output, 1)
print(predict.item())
fig=plt.figure(0)
fig.canvas.set_window_title(str(img_pack[i][1].item()))
wtf=img_pack[i][0][0]
print("wtf.size() = ",wtf.size())
fig=plt.imshow(img_pack[i][0][0].numpy()[0],cmap='gray')


plt.show()


# save_folder="C:\Users\user\Desktop\DontTouchPLSpytorch\Notes\SavedParameters\Cnn1"
# file="Cnn1V1Parameters"
# if(answer):
#     torch.save(myModel.state_dict(), os.path.join(save_folder, file))


#-----------saving parameters-----------
print("enter negative value for exit")
c=0
test_loader_manual=torch.utils.data.DataLoader(dataset=test_dataset)

# print(myModel.parameters())
# print("parameters in first 0 tensor of parameters ", list(myModel.parameters())[0].size())
# print("------------------------------------")
# print("Ere we go 0! : list(myModel.parameters())[0] ", list(myModel.parameters())[0])
# print("------------------------------------")
# print("parameters in second 1 tensor of parameters ", list(myModel.parameters())[1].size())
# print("Ere we go 1! : list(myModel.parameters())[1] ", list(myModel.parameters())[1])

print("end")
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
