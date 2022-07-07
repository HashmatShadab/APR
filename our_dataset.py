import csv
import json
import os
import glob

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets

class_idx = json.load(open("imagenet_class_index.json"))
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
class2label = [class_idx[str(k)][0] for k in range(len(class_idx))]

class_label = {}
label = 0
for name in class2label:
    class_label[name]=label
    label+=1



class ImageNetVal(datasets.ImageFolder):
    def __init__(self, **kwargs):
        super(ImageNetVal, self).__init__(**kwargs)


class ImagesInFolder(Dataset):


    def __init__(self, root_dir, transform):
        super(ImagesInFolder, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.file_names = os.listdir(root_dir)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, item):
        file_name = self.file_names[item]
        file_path = os.path.join(self.root_dir, file_name)
        img = Image.open(file_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        return self.transform(img), file_name

class ImagesInSubFolder(Dataset):


    def __init__(self, root_dir, transform):
        super(ImagesInSubFolder, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.file_paths = glob.glob(f"{root_dir}/**/*.jpg")


    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, item):
        file_path = self.file_paths[item]
        paths = os.path.normpath(file_path)
        paths = paths.split(os.sep)
        file_name, class_name = paths[-1], paths[-2]
        img = Image.open(file_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        return self.transform(img), file_name, class_name





class OUR_dataset(Dataset):
    """
    For a given dataset (data_dir)  and a corresponding csv file (data_csv_dir) with rows as {class name, image names},
    this class create image directory list of the form ['data/path/to/image',label_index] for the first n_Imgs (10) from each class.
    """

    def __init__(self, data_dir:str, data_csv_dir:str, mode:str, img_num:tuple or int, transform):
        assert mode in ['train', 'attack', 'all'], 'WRONG DATASET MODE'
        assert img_num in [1,5,10,20], 'ONLY SUPPORT 2/10/20/40 IMAGES'
        super(OUR_dataset).__init__()
        self.mode = mode
        self.data_dir = data_dir
        data_csv = open(data_csv_dir, 'r')
        csvreader = csv.reader(data_csv)
        data_ls = list(csvreader)
        self.imgs = self.prep_imgs_dir(data_ls, img_num)
        self.transform = transform


    def prep_imgs_dir(self, data_ls, nImg):

        imgs_ls = []
        if self.mode in ['train', 'attack']:
            if nImg>=10:
                sel_ls = list(range(nImg)) #sel_ls=0,...,9
                imgs_ls += self.mk_img_ls(data_ls, sel_ls)
            elif nImg == 1:
                for jkl in list(range(10)):
                    imgs_ls += self.mk_img_ls(data_ls, [jkl])
            elif nImg == 5:
                sel_ls_1 = list(range(5))
                sel_ls_2 = list(range(5,10))
                imgs_ls += self.mk_img_ls(data_ls, sel_ls_1)
                imgs_ls += self.mk_img_ls(data_ls, sel_ls_2)
        elif self.mode == 'all':
            sel_ls = list(range(50))
            imgs_ls += self.mk_img_ls(data_ls, sel_ls)
        return imgs_ls


    def mk_img_ls(self, data_ls, sel_ls):

        """
         Returns the  list (imgs_ls) with ['data/path/to/image',label_index] for the first  n_Imgs  from each row of data_ls
         :param sel_ls: 0,...,(n_Img-1)
        """

        imgs_ls = []
        for label_ind in range(len(data_ls)):
            for img_ind in sel_ls:
                imgs_ls.append([self.data_dir + '/' + data_ls[label_ind][0] + '/' + data_ls[label_ind][1 + img_ind], class_label[data_ls[label_ind][0]]])
        return imgs_ls


    def __getitem__(self, item):
        img = Image.open(self.imgs[item][0])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return self.transform(img), self.imgs[item][1]

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    #One way to prepare 'data/selected_data.csv'
    selected_data_csv = open('data/selected_data_ordered.csv', 'w')
    data_writer = csv.writer(selected_data_csv)
    dataset_dir = 'data/ILSVRC2012_img_val'
    dataset = torchvision.datasets.ImageFolder(dataset_dir)
    label_ind = torch.randperm(500).numpy() # random_ordering : not useful
    label_ind = [i for i in range(500)]
    selected_labels_ls = np.array(dataset.classes)[label_ind]
    for label_name in selected_labels_ls:
        data_writer.writerow([label_name]+os.listdir(os.path.join(dataset_dir, label_name)))
