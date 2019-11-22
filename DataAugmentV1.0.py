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
list_train_in_label_tensors=[]
for index in range(len(names_learn)):
    img_learn_in=Image.open(path_learn_in+names_learn[index])
    img_learn_targets=Image.open(path_learn_targets+names_learn[index])
    img_learn_in=np.array(img_learn_in)
    img_learn_targets=np.array(img_learn_targets)
    size=img_learn_targets.shape
    print("---------BEFORE TRANSFORMATION-----------")
    print("size ", size)
    print(size[0])
    # widht=len(img_learn_targets.data[:,0,0]) # problems because it is a numpy tensor, not a torch tensor
    # for i in range(len(img_learn_targets.data[:,0,0])):
    #     for j in range(len(img_learn_targets.data[0,:,0])):
    for i in range(size[0]):
        for j in range(size[1]):
            if (img_learn_targets.data[i, j, 0] >= 210):
                img_learn_targets.data[i, j, 0] = 1
            else:
                img_learn_targets.data[i, j, 0] = 0
    img_learn_1ch = img_learn_targets[:, :, 0]
    aug=False

    img_learn_in = np.rot90(img_learn_in, 2) # test , deletable
    print("break point")
    if ((random.randint(1,100))>32):
        aug = True
        rotation=random.randint(1,3)
        img_learn_in=np.rot90(img_learn_in,rotation)
        img_learn_1ch=np.rot90(img_learn_1ch,rotation)
    img_learn_targets=torch.from_numpy(img_learn_targets)
    img_learn_in=torch.from_numpy(img_learn_in) # problem
    list_train_in_label_tensors.append((img_learn_in,img_learn_targets))
    path_train_in_save="C:/Users/user/Desktop/DontTouchPLSpytorch/Notes/task2/UnetV1/DataSet1/augmented/train/in/"
    path_train_target_save="C:/Users/user/Desktop/DontTouchPLSpytorch/Notes/task2/UnetV1/DataSet1/augmented/train/targets/"
    if(aug):
        print("---------AFTER Transformation-----------")
        print("img_learn_in.size() ",img_learn_in.size())
        print("img_learn_targets ",img_learn_targets.size())
        torch.save(img_learn_in,path_train_in_save+"aug"+names_learn[index]) # will be save as *tiff !!
        torch.save(img_learn_targets,path_train_target_save+"aug"+names_learn[index]) # will be save as *tiff !!
    else:
        pass
#----------------------------------------------


img_learn_targets=torch.from_numpy(img_learn_targets)
plt.imshow((np.array(img_learn_1ch)),cmap='gray')

print("break point")

class ValidData(data.Dataset):
    def __init__(self):
        pass
    def __getitem__(self, item):
        img=np.array(Image.open(path_vaild_in+names_valid[item]))
        label=np.array(Image.open(path_vaild_targets+names_valid[item]))


