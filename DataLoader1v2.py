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

train_dataset=MyDataset()
#--------------hyper parameters------------
batch_size=4
num_workers=4
#--------------hyper parameters------------
def detection_collate(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(torch.FloatTensor(sample[0]).permute(2, 0, 1))
        # targets.append(torch.FloatTensor([[sample[1]]]))
        targets.append(torch.FloatTensor(sample[1].permute(2,0,1)))

    return torch.stack(imgs, 0), targets

data_train_loader=data.DataLoader(train_dataset, batch_size,num_workers=num_workers,shuffle=True,collate_fn=detection_collate,pin_memory=True,drop_last=True)







print("hello")
