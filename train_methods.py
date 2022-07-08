import sys
import time

import torchvision
import wandb
import torch.nn.functional as F
from fgsm import FGSM
from utils import *


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
