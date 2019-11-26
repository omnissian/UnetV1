
#--------------amount of classes = 2-------------
#---------below--dataloaders for cross entropy loss-----each class have his own slice(dimension) of disribution of probability


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



#---above--dataloaders for cross entropy loss-----each class have his own slice(dimension) of disribution of probability
