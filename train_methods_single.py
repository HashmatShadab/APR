import torchvision

import wandb
from model import *
from utils import *
from fgsm import FGSM
import os

import numpy as np
import sys


def plot_grid(w):
    import matplotlib.pyplot as plt
    grid_img = torchvision.utils.make_grid(w)
    plt.imshow(grid_img.permute(1,2,0).cpu())
    plt.show()


def train_unsup(model, data_loader,optimizer,args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    for epoch in range(args.start_epoch, args.end_epoch):
        avg_batch_loss_epoch = 0
        running_loss = 0

        for idx, (img, _) in enumerate(data_loader):
            img =img.to(device)
            img_input = img
            img_tar = img.clone()
            for img_ind in range(img_input.shape[0]):
                if args.mode == 'rotate':
                    img_input[img_ind:img_ind + 1] = rot(img_input[img_ind:img_ind + 1])
                elif args.mode == 'jigsaw':
                    img_input[img_ind] = shuffle(img_input[img_ind], 1)
                else:
                    sys.exit("Enter the correct mode")

            outputs, _, _ = model(img_input)
            loss = nn.MSELoss()(outputs[0], img_tar)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_batch_loss_epoch += loss.item()
            if idx % 10 == 9:
                print('Epoch: {0} \t Batch: {1} \t loss: {2:.5f}'.format(epoch, idx, running_loss / 10))
                wandb.log({"Running Loss": running_loss / 10})
                running_loss = 0

            running_loss += abs(loss.item())
            length = len(data_loader) // 2
            if (idx + 1) % length == 0:
                rand_ind = torch.randint(0, img_input.shape[0], size=(1,))
                outputs = outputs[0]

                wandb.log(
                    {'Image': [wandb.Image(img_input[rand_ind.item()].permute(1, 2, 0).cpu().numpy(), caption="Input"),
                               wandb.Image(img_tar[rand_ind.item()].permute(1, 2, 0).cpu().numpy(), caption="Target"),
                               wandb.Image(outputs[rand_ind.item()].permute(1, 2, 0).detach().cpu().numpy(),
                                           caption="Output")],
                     })


        print(f"Epoch {epoch}: Average Loss on Batch::: {avg_batch_loss_epoch / (len(data_loader))}")
        wandb.log({"Loss": avg_batch_loss_epoch / (len(data_loader))})
        os.makedirs(args.save_dir + f'/models', exist_ok=True)


        if (epoch + 1) % args.save_epoch == 0:
            model.eval()
            torch.save(model.state_dict(), args.save_dir + f'/models/{args.mode}_{epoch}.pth')
            model.train()



def train_unsup_adv(model, data_loader,optimizer,args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    fgsm_step = args.fgsm_step/255


    attack = FGSM(model, eps=fgsm_step)

    for epoch in range(args.start_epoch, args.end_epoch):
        avg_batch_loss_epoch = 0
        running_loss = 0
        for idx, (img, _) in enumerate(data_loader):
            img =img.to(device)
            img_input = img
            img_tar = img.clone()
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


            avg_batch_loss_epoch += loss.item()
            if idx % 10 == 9:
                print('Epoch: {0} \t Batch: {1} \t sim_loss: {2:.5f}\t running_loss: {3:.5f}'.format(epoch, idx, sim_loss.item(), running_loss / 10))

                wandb.log({"Running Loss": running_loss / 10})
                running_loss = 0

            running_loss += abs(loss.item())

            length = len(data_loader) // 2
            if (idx + 1) % length == 0:
                rand_ind = torch.randint(0, img_input.shape[0], size=(1,))
                wandb.log(
                    {'Image': [
                        wandb.Image(img_input[rand_ind.item()].permute(1, 2, 0).cpu().numpy(), caption="Input-Clean"),
                        wandb.Image(adv_images[rand_ind.item()].permute(1, 2, 0).cpu().numpy(), caption="Input-Adv"),
                        ],
                     })

        print(f"Epoch {epoch}: Average Loss on Batch::: {avg_batch_loss_epoch / (len(data_loader))}")
        wandb.log({"Loss": avg_batch_loss_epoch / (len(data_loader))})
        os.makedirs(args.save_dir + f'/models', exist_ok=True)

        if (epoch + 1) % args.save_epoch == 0:
            model.eval()
            torch.save(model.state_dict(), args.save_dir + f'/models/{args.mode}_{epoch}.pth')
            model.train()
