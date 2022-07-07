import csv
import os

import pretrainedmodels
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from einops import rearrange
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from ImagenetDataset import ImageFolder
from transformers import diet_tiny, diet_small, vit_tiny, vit_small


# Normalization
class Normalize(nn.Module):
    def __init__(self, ms=None):
        super(Normalize, self).__init__()
        if ms == None:
            self.ms = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]

    def forward(self, input):
        x = input.clone()
        for i in range(x.shape[1]):
            x[:, i] = (x[:, i] - self.ms[0][i]) / self.ms[1][i]
        return x


def mask(img, mask_ratio, patch_size):
    """
    To create masking patches in the img given a mask-ratio
    :param mask_ratio: Image area to be masked (between 0-1)
    :param patch_size: Size of the image patches.
    :return: returns the masked image
    """
    patches = patchify(img, patch_size=patch_size)
    N, L, D = patches.shape
    len_mask = int(L * (mask_ratio))
    noise = torch.rand(N, L)  # noise in [0, 1]
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    patches[:, ids_shuffle[0][0:len_mask], :] = 0
    masked_img = unpatchify(patches, patch_size=patch_size)
    masked_img = masked_img
    return masked_img


def rot(img):
    """
    To rotate the image randomly in one of the four angles (0,90,180 & 270)
    :return: returns rotated img.
    """
    rand_angle = torch.randint(0, 4, size=(1,)).item()  # 0,1,2,3
    assert rand_angle in [0, 1, 2, 3], 'check rand_angle'
    if rand_angle == 0:
        img = img
    elif rand_angle == 1:
        img = torch.flip(img, dims=[3]).permute(0, 1, 3, 2)
    elif rand_angle == 2:
        img = torch.flip(img, dims=[2])
        img = torch.flip(img, dims=[3])
    elif rand_angle == 3:
        img = torch.flip(img.permute(0, 1, 3, 2), dims=[3])
    return img


def apply_2d_rotation(input_tensor, rot=0):
    """Apply a 2d rotation of 0, 90, 180, or 270 degrees to a tensor.
    The code assumes that the spatial dimensions are the last two dimensions,
    e.g., for a 4D tensors, the height dimension is the 3rd one, and the width
    dimension is the 4th one.
    """

    assert input_tensor.dim() >= 2

    height_dim = input_tensor.dim() - 2
    width_dim = height_dim + 1
    r = rot
    rotation = r * 90
    flip_upside_down = lambda x: torch.flip(x, dims=(height_dim,))
    flip_left_right = lambda x: torch.flip(x, dims=(width_dim,))
    spatial_transpose = lambda x: torch.transpose(x, height_dim, width_dim)

    if rotation == 0:  # 0 degrees rotation
        return input_tensor
    elif rotation == 90:  # 90 degrees rotation
        return flip_upside_down(spatial_transpose(input_tensor))
    elif rotation == 180:  # 90 degrees rotation
        return flip_left_right(flip_upside_down(input_tensor))
    elif rotation == 270:  # 270 degrees rotation / or -90
        return spatial_transpose(flip_upside_down(input_tensor))
    else:
        raise ValueError(
            "rotation should be 0, 90, 180, or 270 degrees; input value {}".format(rotation)
        )


def horizontal_flip(img):
    rand_flip = torch.randint(0, 2, size=(1,)).item()  # 0,1
    assert rand_flip in [0, 1], 'check rand_flip'
    img = torch.flip(img, dims=[3])
    return img


def shuffle(img, mode):
    """
    To create 4 tile jigsaw patches of the given image
    :param mode: Shuffle mode (1)
    :return: returns shuffled img.
    """
    assert mode in [0, 1], 'check shuffle mode'
    if mode == 0:
        patch_0 = img[:, 0:112, 0:112]
        patch_1 = img[:, 0:112, 112:224]
        patch_2 = img[:, 112:224, 0:112]
        patch_3 = img[:, 112:224, 112:224]
        rand_ind = torch.randperm(4)
        img_0 = torch.cat((eval('patch_{}'.format(rand_ind[0])),
                           eval('patch_{}'.format(rand_ind[1]))), dim=2)
        img_1 = torch.cat((eval('patch_{}'.format(rand_ind[2])),
                           eval('patch_{}'.format(rand_ind[3]))), dim=2)
        return torch.cat((img_0, img_1), dim=1)
    else:
        # four possibilities, for easy training
        img = img.permute(1, 2, 0)
        img = img.reshape(2, 112, 224, 3)
        rand_shuffle_1 = torch.randint(0, 2, size=(1,)).item()
        img = img[[rand_shuffle_1, 1 - rand_shuffle_1]]
        img = img.reshape(224, 224, 3)
        img = img.permute(1, 0, 2)
        img = img.reshape(2, 112, 224, 3)
        rand_shuffle_2 = torch.randint(0, 2, size=(1,)).item()
        img = img[[rand_shuffle_2, 1 - rand_shuffle_2]]
        img = img.reshape(224, 224, 3)
        return img.permute(2, 1, 0)


def aug(img_input):
    for img_ind in range(img_input.shape[0]):
        img_input[img_ind:img_ind + 1] = horizontal_flip(img_input[img_ind:img_ind + 1])
    return img_input


def mk_proto_ls(n_imgs):
    """
      To create list of prototypes for prototypical reconstruction approach
      :param n_imgs: Number of reference images (10)
      :return: returns a list of 100 pairs in which the first 10 pairs are
      [0,10],[1,11],...,[9,19] and the rest are random.
    """
    tar_ind_ls = torch.tensor(list(range(int(2 * n_imgs)))).reshape((2, n_imgs)).permute((1, 0)).reshape(-1)
    # [0, 10, 1, 11, .........., 8, 18, 9, 19]
    tar_ind_ls_ex = []
    for i_f in list(range(n_imgs)):
        for i_s in list(range(n_imgs, n_imgs * 2)):
            if i_f != i_s - n_imgs:
                tar_ind_ls_ex.append([i_f, i_s])
    # [[0, 11], [0, 12], [0, 13],.......[0,19],
    #  [1, 10], [1, 12], [1, 13],.......[1,19],
    #    .
    #    .
    #  [9, 10], [9, 11], [9, 12],.......[9, 18]] shape > (90, 2)

    # randomly order the 90 pairs and then put them in single array
    tar_ind_ls_ex = torch.tensor(tar_ind_ls_ex)[torch.randperm(len(tar_ind_ls_ex))].reshape(-1)
    # add [0, 10, 1, 11, .........., 8, 18, 9, 19] back to the randomly ordered 90 pairs
    tar_ind_ls = torch.cat((tar_ind_ls, tar_ind_ls_ex))
    return tar_ind_ls


def patchify(imgs, patch_size):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    p = patch_size
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
    return x


def unpatchify(x, patch_size):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    p = patch_size
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs


def input_diversity(img):
    rnd = torch.randint(224, 257, (1,)).item()
    rescaled = F.interpolate(img, (rnd, rnd), mode='nearest')
    h_rem = 256 - rnd
    w_hem = 256 - rnd
    pad_top = torch.randint(0, h_rem + 1, (1,)).item()
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_hem + 1, (1,)).item()
    pad_right = w_hem - pad_left
    padded = F.pad(rescaled, pad=(pad_left, pad_right, pad_top, pad_bottom))
    padded = F.interpolate(padded, (224, 224), mode='nearest')
    return padded




def classify(save_dir, batch_size, save_results):
    from torchvision import models
    image_transforms_adv = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    data = ImageFolder(root=save_dir, transform=image_transforms_adv)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = {"Resnet-152": models.resnet152, "VGG-19": models.vgg19_bn, "Inception-V3": models.inception_v3,
              "DenseNet-161": models.densenet161, "DenseNet-121": models.densenet121,
              "WRN-101": models.wide_resnet101_2, "MobileNet-v2": models.mobilenet_v2,
              "senet": pretrainedmodels.__dict__['senet154']}

    model_results_csv = open(f'{os.path.join(save_dir, save_results)}.csv', 'w')  # append?
    data_writer = csv.writer(model_results_csv)
    title = ['image_type', save_dir]
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
                print(labels)
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

    transformers = {"diet_tiny": diet_tiny, "diet_small": diet_small, "vit_tiny": vit_tiny, "vit_small": vit_small}
    for name, transformer in transformers.items():
        model = transformer()
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
        transform.transforms.pop()
        transform_clean = transform
        transform_adv = transforms.Compose([transforms.ToTensor(), ])
        transform = transform_adv if adv else transform_clean
        norm_layer = Normalize(mean=config['mean'],
                               std=config['std'])
        model = nn.Sequential(norm_layer, model.to(device=device))
        data = ImageFolder(root=save_dir, transform=transform)
        test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)

        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for (images, labels) in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                print(labels)
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


def mask_batch(images, mask_ratio, patch_size):
    patches = patchify(images, patch_size=patch_size)
    num_masked = int(mask_ratio * patches.shape[1])
    masked_indices = torch.rand(patches.shape[0], patches.shape[1]).topk(k=num_masked, dim=-1).indices
    masked_bool_mask = torch.zeros((patches.shape[0], patches.shape[1])).scatter_(-1, masked_indices, 1).bool()
    mask_patches = torch.zeros_like(patches)
    patches = torch.where(masked_bool_mask[..., None], mask_patches, patches)
    unpatches = unpatchify(patches, patch_size=patch_size)

    return unpatches


import numpy as np
from scipy import stats as st


def gkern(kernlen=15, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


def ukern(kernlen=15):
    kernel = np.ones((kernlen, kernlen)) * 1.0 / (kernlen * kernlen)
    return kernel


def lkern(kernlen=15):
    kern1d = 1 - np.abs(np.linspace((-kernlen + 1) / 2, (kernlen - 1) / 2, kernlen) / (kernlen + 1) * 2)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


def kernel(len_kernel=15, nsig=3, kernel_name="gaussian"):
    if kernel_name == 'gaussian':
        kernel = gkern(len_kernel, nsig).astype(np.float32)
    elif kernel_name == 'linear':
        kernel = lkern(len_kernel).astype(np.float32)
    elif kernel_name == 'uniform':
        kernel = ukern(len_kernel).astype(np.float32)
    else:
        raise NotImplementedError

    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 1)
    return stack_kernel


def shuffle_einop(img, shuffle_size, label):
    assert img.shape[2] % shuffle_size == 0, f"shuffle size {shuffle_size} not" \
                                             f"compatible with {img.shape[2]} image"
    labels = []
    patch_dim1, patch_dim2 = img.shape[2] // shuffle_size, img.shape[3] // shuffle_size
    patch_num = shuffle_size * shuffle_size
    img = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_dim1, p2=patch_dim2)

    for i in range(img.shape[0]):
        # row = np.random.choice(range(patch_num), size=img.shape[1], replace=False)
        row = label
        img[i:i + 1] = img[i:i + 1, row, :]

    img = rearrange(img, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                    h=shuffle_size, w=shuffle_size, p1=patch_dim1, p2=patch_dim2)

    return img


