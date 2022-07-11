import argparse
import csv
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import wandb
from torch.backends import cudnn
from torch.utils.data import Dataset

from fgsm import FGSM
from surrogate import *
from our_dataset import OUR_dataset
from utils import *

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--project', type=str, default="APR")
parser.add_argument('--entity', type=str, default="hashmatshadab")
parser.add_argument('--wandb_mode', type=str, default="disabled")
parser.add_argument('--n_imgs', type=int, default=20, help='number of all reference images')
parser.add_argument('--n_iters', type=int, default=2000)
parser.add_argument('--n_decoders', type=int, default=20)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--mode', type=str, default='rotate')
parser.add_argument('--save_dir', type=str, default='./trained_models')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=2500)
parser.add_argument('--fgsm_step', type=int, default=2)
parser.add_argument('--adv_train', type=lambda x: (str(x).lower() == 'true'), default=False)



def create_json(args):
    """
     To create json file to save the arguments (args)
    """

    with open(f"{args.save_dir}/config_train.json", "w") as write_file:
        json.dump(args.__dict__, write_file, indent=4)

def initialize_model(decoder_num):
    """
        Initialize the auto-encoder model with given number of decoders
        :param decoder_num: Number of decoders (20 for prototypical and 1 for other modes)
    """
    model = autoencoder(input_nc=3, output_nc=3, n_blocks=3, decoder_num=decoder_num)
    model = nn.Sequential(
        Normalize(),
        model,
    )
    model.to(device)
    return model


def plot_grid(w):
    import matplotlib.pyplot as plt
    grid_img = torchvision.utils.make_grid(w)
    plt.imshow(grid_img.permute(1,2,0).cpu())
    plt.show()


def train_prototypical(model, img, n_imgs, n_decoders, n_iters, prototype_ind_csv_writer,optimizer, train_loss, iter_ind):
    """
    Training using the prototypical reconstruction approach

    :param img: Images to train each substitute model ( 10 images each from 2 classes) [20 x 3 x 224 x 224]
    :param n_imgs: Number of  images (10 from each class)
    :param n_decoders: Number of decoders (20) [20 x 56 x 56]
    :param n_iters:  Number of iterations for training each substitute model
    :param prototype_ind_csv_writer: To write the prototype pairs for the given autoencoder
    :param train_loss: dictionary to save loss at each iteration
    :return: returns the substitute model trained on images from 2 classes
    """
    do_aug = True
    if n_imgs == 1:
        tar_ind_ls = [0, 1]
    else:
        tar_ind_ls = mk_proto_ls(n_imgs)
    tar_ind_ls = tar_ind_ls[:n_decoders * 2]
    # get the first 20 pairs (first 10 pairs are same always [0, 10, 1, 11, .........., 8, 18, 9, 19])
    prototype_ind_csv_writer.writerow(tar_ind_ls.tolist()) # save the 20 pairs used in this model
    img_tar = img[tar_ind_ls]
    if n_decoders != 1:
        img_tar = F.interpolate(img_tar, (56, 56))
    since = time.time()
    train_loss[iter_ind] = []
    for i in range(n_iters):
        rand_ind = torch.cat((torch.randint(0, n_imgs, size=(1,)), torch.randint(n_imgs, 2 * n_imgs, size=(1,))))
        # get random indices of the 2 images to be chosen from two different classes
        img_input = img[rand_ind].clone()
        if do_aug:
            img_input = aug(img_input)
        assert img_input.shape[3] == 224
        outputs, _,_ = model(img_input)
        gen_img = torch.cat(outputs, dim=0)
        loss = nn.MSELoss()(gen_img, img_tar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss[iter_ind].append(loss.item())

        if (i + 1) % 500 == 0:
            print(iter_ind + 1, i + 1, round(loss.item(), 5), '{} s'.format(int(time.time() - since)))
            rand_ind = torch.randint(0, img_input.shape[0], size=(1,))
            wandb.log(
                {'Image': [wandb.Image(img_input[rand_ind.item()].permute(1, 2, 0).cpu().numpy(), caption="Input"),
                           wandb.Image(img_tar[rand_ind.item()].permute(1, 2, 0).cpu().numpy(), caption="Target"),
                           wandb.Image(outputs[0][rand_ind.item()].permute(1, 2, 0).detach().cpu().numpy(),
                                       caption="Output")],
                 })
    return model



def train_adv_prototypical(model, img, n_imgs, n_decoders, n_iters,prototype_ind_csv_writer, optimizer,train_loss, iter_ind, fgsm_step):

    """
    Training using prototypical reconstruction approach incorporated with free adversarial training

    :param img:  Images to train each substitute model ( default: 10 images each from 2 classes) [20 x 3 x 224 x 224]
    :param n_imgs: Number of reference images (10 from each class)
    :param n_decoders: Number of decoders (default 20) [output size 56 x 56]
    :param n_iters:  Number of iterations for training each substitute model
    :param prototype_ind_csv_writer: To write the prototype pairs for the given autoencoder
    :param train_loss: dictionary for saving loss for each autoencoder
    :param fgsm_step: perturbation budget for generating adversarial example
    :return: returns the substitute model
    """


    do_aug = True
    if n_imgs == 1:
        tar_ind_ls = [0, 1]
    else:
        tar_ind_ls = mk_proto_ls(n_imgs)
    tar_ind_ls = tar_ind_ls[:n_decoders * 2]
    # get the first 20 pairs (first 10 pairs are same always [0, 10, 1, 11, .........., 8, 18, 9, 19])
    prototype_ind_csv_writer.writerow(tar_ind_ls.tolist())  # save the 20 pairs used in this model


    img_tar = img[tar_ind_ls]
    if n_decoders != 1:
        img_tar = F.interpolate(img_tar, (56, 56))
    since = time.time()
    train_loss[iter_ind] = []
    attack = FGSM(model, eps=fgsm_step, mode="proto")
    for i in range(n_iters):
        rand_ind = torch.cat((torch.randint(0, n_imgs, size=(1,)), torch.randint(n_imgs, 2 * n_imgs, size=(1,))))
        # get random indices of the 2 images to be chosen from two different classes
        img_input = img[rand_ind].clone()
        if do_aug:
            img_input = aug(img_input)
        assert img_input.shape[3] == 224
        adv_images = attack(img_input, img_tar)


        clean_output, clean_enc, _ = model(img_input)
        clean_output = torch.cat(clean_output, dim=0)
        adv_output, adv_enc, _ = model(adv_images)
        adv_output = torch.cat(adv_output, dim=0)
        loss_clean = nn.MSELoss()(clean_output, img_tar)
        loss_adv = nn.MSELoss()(adv_output, img_tar)
        sim_loss = nn.MSELoss()(adv_enc, clean_enc)


        loss = loss_clean + loss_adv + sim_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss[iter_ind].append(loss.item())

        if (i + 1) % 500 == 0:
            print(
                f"{iter_ind + 1}, {i + 1}, Total Loss {round(loss.item(), 5)}, Sim Loss {round(sim_loss.item(), 5)}, {int(time.time() - since)} s")
            rand_ind = torch.randint(0, img_input.shape[0], size=(1,))
            wandb.log(
                {'Image': [wandb.Image(img_input[rand_ind.item()].permute(1, 2, 0).cpu().numpy(), caption="Input-Clean"),
                           wandb.Image(adv_images[rand_ind.item()].permute(1, 2, 0).cpu().numpy(), caption="Input-Adv"),
                           ],
                 })

    return model

def train_unsup(model, img, n_iters,optimizer,args, train_loss, iter_ind):
    """
    Training using self supervised (rotation/jigsaw/masking) approach

    :param img: Images to train each substitute model ( 10 images each from 2 classes) [20 x 3 x 224 x 224]
    :param n_iters: Number of iterations for training each substitute model
    :param train_loss: dictionary for storing the loss for each autoencoder
    :param iter_ind: the corresponding batch number on which autoencoder is being trained
    :return: returns the substitute model trained on images from 2 classes
    """
    img_input = img
    img_tar = img.clone()
    since = time.time()
    train_loss[iter_ind] = []
    for i in range(n_iters):
        for img_ind in range(img_input.shape[0]):
            if args.mode == 'rotate':
                img_input[img_ind:img_ind + 1] = rot(img_input[img_ind:img_ind + 1])
            elif args.mode == 'jigsaw':
                img_input[img_ind] = shuffle(img_input[img_ind], 1)
            else:
                sys.exit("Enter the correct mode")


        outputs, _,_ = model(img_input)
        loss = nn.MSELoss()(outputs[0], img_tar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss[iter_ind].append(loss.item())
        if (i + 1) % 500 == 0:
            print(iter_ind + 1, i + 1, round(loss.item(), 5), '{} s'.format(int(time.time() - since)))
            rand_ind = torch.randint(0, img_input.shape[0], size=(1,))
            wandb.log(
                {'Image': [wandb.Image(img_input[rand_ind.item()].permute(1, 2, 0).cpu().numpy(), caption="Input"),
                           wandb.Image(img_tar[rand_ind.item()].permute(1, 2, 0).cpu().numpy(), caption="Target"),
                           wandb.Image(outputs[0][rand_ind.item()].permute(1, 2, 0).detach().cpu().numpy(),
                                       caption="Output")],
                 })

    return model


def train_adv_unsup(model, img, n_iters, optimizer, args,train_loss, iter_ind, fgsm_step):
    """
    Training using self supervised (rotation/jigsaw/masking) approach incorporated with free adversarial training

    :param img: Images to train each substitute model ( 10 images each from 2 classes) [20 x 3 x 224 x 224]
    :param n_iters: Number of iterations for training each substitute model
    :param train_loss: dictionary for storing the loss for each autoencoder
    :param iter_ind: the corresponding batch number on which autoencoder is being trained
    :param fgsm_step: perturbation budget for generating adversarial example
    :return: returns the substitute model
    """
    img_input = img
    img_tar = img.clone()
    since = time.time()
    train_loss[iter_ind] = []
    attack = FGSM(model, eps=fgsm_step)

    for i in range(n_iters):
        for img_ind in range(img_input.shape[0]):
            if args.mode == 'rotate':
                img_input[img_ind:img_ind + 1] = rot(img_input[img_ind:img_ind + 1])
            elif args.mode == 'jigsaw':
                img_input[img_ind] = shuffle(img_input[img_ind], 1)
            else:
                sys.exit("Enter the correct mode")


        adv_images = attack(img_input, img_tar)

        clean_output, clean_enc, _ = model(img_input)
        adv_output, adv_enc, _ = model(adv_images)
        loss_clean = nn.MSELoss()(clean_output[0], img_tar)
        loss_adv = nn.MSELoss()(adv_output[0], img_tar)
        sim_loss = nn.MSELoss()(adv_enc, clean_enc)

        loss = loss_clean + loss_adv + sim_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss[iter_ind].append(loss.item())
        if (i + 1) % 500 == 0:
            print(f"{iter_ind + 1}, {i+1}, Total Loss {round(loss.item(), 5)}, Sim Loss {round(sim_loss.item(), 5)}, {int(time.time() - since)} s")
            rand_ind = torch.randint(0, img_input.shape[0], size=(1,))
            wandb.log(
                {'Image': [wandb.Image(img_input[rand_ind.item()].permute(1, 2, 0).cpu().numpy(), caption="Input-Clean"),
                           wandb.Image(adv_images[rand_ind.item()].permute(1, 2, 0).cpu().numpy(), caption="Input-Adv"),
                           ],
                 })

    return model

if __name__ == '__main__':

    """
        Training of multiple autoencoders using rotation, jigsaw, and prototypical methods.
        Default : Each autoencoder is trained on only 20 images (10 from each class) using the above methods.
        In total 250 autoencoders are trained on a subset 0f 5000 images (10 from each class) from ImageNetval.
    """

    args = parser.parse_args()
    wandb.init(project=args.project, entity=args.entity, mode=args.wandb_mode, name=args.save_dir.split("/")[-1])

    cudnn.benchmark = False
    cudnn.deterministic = True
    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    print(args)
    config = wandb.config
    config.update(args)

    mode = args.mode
    assert mode in ['prototypical', 'unsup_naive', 'jigsaw', 'rotate', 'mask']
    save_dir = args.save_dir
    n_imgs = args.n_imgs // 2
    n_iters = args.n_iters
    lr = args.lr
    # only in prototypical methods there are more than 1 decoders in the autoencoder
    if mode != 'prototypical':
        n_decoders = 1
    else:
        n_decoders = args.n_decoders
    assert n_decoders <= n_imgs**2, 'Too many decoders.'

    config.update({'n_decoders': n_decoders}, allow_val_change=True)

    os.makedirs(save_dir+'/models', exist_ok=True) # create directory to save the trained models
    create_json(args)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    batch_size = n_imgs*2
    config.update({'batch_size': batch_size})
    do_aug = True

    trans = T.Compose([                             # Normalisation of images is done in the first layer of the autoencoder
        T.Resize((256,256)),
        T.CenterCrop(224),
        T.ToTensor()
    ])

    dataset = OUR_dataset(data_dir='data/ILSVRC2012_img_val',
                          data_csv_dir='data/selected_data.csv',  # from the selected 500 classes of ImageNet val, the dataset loads the first 10 images from each class.
                          mode='train',
                          img_num=n_imgs,
                          transform=trans,
                          )

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers = 1)


    if mode == 'prototypical':
        prototype_ind_csv = open(save_dir+'/prototype_ind.csv', 'a', newline='') # create a csv file to save prototypes used during the training of each autoencoder.
        prototype_ind_csv_writer = csv.writer(prototype_ind_csv)

    fig, ax = plt.subplots()
    for iter_ind, (img, label_ind) in enumerate(data_loader):

        if not args.start <= iter_ind < args.end:
            continue

        model = initialize_model(n_decoders)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        img = img.to(device)


        train_loss = {}
        if mode == 'prototypical':
            if args.adv_train:

                train_adv_prototypical(model, img, n_imgs, n_decoders, n_iters, prototype_ind_csv_writer=prototype_ind_csv_writer,
                                       optimizer=optimizer, train_loss=train_loss, iter_ind=iter_ind, fgsm_step=args.fgsm_step / 255.0)
            else:
                train_prototypical(model, img, n_imgs, n_decoders, n_iters, prototype_ind_csv_writer=prototype_ind_csv_writer,
                                   optimizer=optimizer, train_loss=train_loss, iter_ind=iter_ind)


        else:

            if args.adv_train:
                train_adv_unsup(model, img, n_iters, optimizer, args, train_loss=train_loss, iter_ind=iter_ind,fgsm_step=args.fgsm_step / 255.0)
            else:
                train_unsup(model, img, n_iters, optimizer, args, train_loss=train_loss, iter_ind=iter_ind)

        model.eval()

        torch.save(model.state_dict(), save_dir + f'/models/{args.mode}_{iter_ind}.pth')

        xs = [x for x in range(len(train_loss[iter_ind]))]
        ax.plot(xs, train_loss[iter_ind], label=f"Model_{iter_ind}")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Loss")
        ax.set_title(f"Model_{mode}_{iter_ind}")
        wandb.log({f'plot': ax})

