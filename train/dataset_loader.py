import os
from torchvision.io import read_image
from torch.utils.data import Dataset
import cv2
import random
import torch

class ImageClassificationDataset(Dataset):
    def __init__(self, img_source, transform=None, target_transform=None, labels=None):
        self.transform = transform
        self.target_transform = target_transform
        if type(img_source) == str:
            self._load_imagens_from_dir(img_source)
        else:
            if labels is None:
                raise(Exception("labels needed to construct Dataset"))
            self._load_from_image_list(img_source, labels)



    def _load_imagens_from_dir(self, img_dir):

        self.img_list = []
        self.labels = []

        for file in os.listdir(img_dir):
            path = os.path.join(img_dir, file)
            if os.path.isdir(path):
                self.labels.append(file)
        
        for label in self.labels:
            label_path = os.path.join(img_dir, label)
            for file in os.listdir(label_path):
                path = os.path.join(label_path,file)
                ext = os.path.splitext(file)[1]
                if ext == ".png" or ext == ".jpeg" or ext == ".jpg":
                    self.img_list.append([path,label])

    def _load_from_image_list(self, img_list, labels):
        self.img_list = img_list
        self.labels = labels

    def __len__(self):
        return len(self.img_list)

    
    def __getitem__(self, index):
        img_path, label_str = self.img_list[index]
        image = cv2.imread(img_path)
        label = self.labels.index(label_str)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
    def split(self, proportion=0.8, seed=4444):
        random.seed(seed)
        train = []
        test = []

        for img in self.img_list:
            if random.randint(0,100)/100 < proportion:
                train.append(img.copy())
            else:
                test.append(img.copy())
        
        trainDataset = ImageClassificationDataset(train, self.transform, self.target_transform, self.labels)
        testDataset = ImageClassificationDataset(test, self.transform, self.target_transform, self.labels)
        return trainDataset, testDataset
        

if __name__=="__main__":
    dataset = ImageClassificationDataset("./datasets/Rice_Image_Dataset")
    
    train, test = dataset.split()
    img, label = test[-1]

    print(len(train))
    print(len(test))
    print(len(dataset))
    print(img.shape)
    print(label)
    cv2.imshow("img", img)
    cv2.waitKey()