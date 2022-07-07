import argparse
import csv
import os

import pretrainedmodels
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from PIL import ImageFile
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from transformers import diet_tiny, diet_small, vit_tiny, vit_small

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        # Broadcasting
        if torch.max(input) > 1:
            input = input / 255.0
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean.to(device=input.device)) / std.to(
            device=input.device)


def classify(save_dir, batch_size, save_results, adv=True):

    image_transforms_adv = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    image_transfroms_clean = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

    transform = image_transforms_adv if adv else image_transfroms_clean


    data = torchvision.datasets.folder.ImageFolder(root=save_dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = {"Resnet-152": torchvision.models.resnet152, "VGG-19": torchvision.models.vgg19_bn, "Inception-V3": torchvision.models.inception_v3,
              "DenseNet-161": torchvision.models.densenet161, "DenseNet-121": torchvision.models.densenet121,
              "WRN-101": torchvision.models.wide_resnet101_2, "MobileNet-v2": torchvision.models.mobilenet_v2,
              "senet": pretrainedmodels.__dict__['senet154']}

    model_results_csv = open(f'{os.path.join(save_dir, save_results)}.csv', 'w')  # append?
    data_writer = csv.writer(model_results_csv)
    title = ['image_type',save_dir]
    data_writer.writerow(title)
    header = ['model', 'Accuracy']
    data_writer.writerow(header)
    avg_accuracy = 0
    for name, obj in models.items():
        if name == "senet":
            model = obj(num_classes=1000, pretrained='imagenet')
        else:
            model = obj(pretrained=True)

        model.to(device)
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for (images, labels) in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                print(total, end="\r")
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the model {name} on the test images: {100 * correct / total} %')
        accuracy = 100 * correct / total
        avg_accuracy += accuracy
        data_writer.writerow([name, accuracy])
    print(f'Average accuracy on models {avg_accuracy / len(models.items())} %')
    print(f"Results saved in {os.path.join(save_dir, save_results)}.csv")
    data_writer.writerow(["Average accuracy", avg_accuracy / len(models.items())])



def classifiy_transformers(save_dir, batch_size, save_results, adv=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_results_csv = open(f'{os.path.join(save_dir, save_results)}_transformers.csv', 'w')  # append?
    data_writer = csv.writer(model_results_csv)
    title = ['image_type', save_dir]
    data_writer.writerow(title)
    header = ['model', 'Accuracy']
    data_writer.writerow(header)
    avg_accuracy = 0

    transformers =  {"diet_tiny":diet_tiny, "diet_small":diet_small, "vit_tiny":vit_tiny,"vit_small":vit_small}
    for name, transformer in transformers.items():
        model = transformer()
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
        transform.transforms.pop()
        transform_clean = transform
        transform_adv =  transforms.Compose([transforms.ToTensor(), ])
        transform = transform_adv if adv else transform_clean
        norm_layer = Normalize(mean=config['mean'],
                               std=config['std'])
        model = nn.Sequential(norm_layer, model.to(device=device))
        data = torchvision.datasets.folder.ImageFolder(root=save_dir, transform=transform)
        test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)



        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for (images, labels) in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                print(total, end="\r")
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the model {name} on the test images: {100 * correct / total} %')
        accuracy = 100 * correct / total
        avg_accuracy += accuracy
        data_writer.writerow([name, accuracy])
    print(f'Average accuracy on models {avg_accuracy / len(transformers.items())} %')
    print(f"Results saved in {os.path.join(save_dir, save_results)}_transformers.csv")
    data_writer.writerow(["Average accuracy", avg_accuracy / len(transformers.items())])

parser = argparse.ArgumentParser(description='Classification')
parser.add_argument('--data_path', type=str, default='adv_images_rotate')
parser.add_argument('--mode', type=str, default='adv')
parser.add_argument('--test_model', type=str, default='all')
parser.add_argument('--save_results', type=str, default='results_')
parser.add_argument('--batch_size', type=int, default=64)
args = parser.parse_args()

if __name__ == "__main__":
    mode = args.mode == "adv"
    classify(save_dir=args.data_path, batch_size=args.batch_size, save_results=args.save_results,adv= mode )
    classifiy_transformers(save_dir=args.data_path, batch_size=args.batch_size, save_results=args.save_results, adv=mode)













