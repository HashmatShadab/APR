import argparse
import json
import os
import sys

import PIL
import numpy as np
import torchvision
import torchvision.datasets.folder
import wandb
from PIL import Image
from PIL import ImageFile
from torch.backends import cudnn
from torchvision import transforms as T

from surrogate import *
from utils import *

PIL.Image.MAX_IMAGE_PIXELS = 933120000

ImageFile.LOAD_TRUNCATED_IMAGES = True

def plot_grid(w):
    import matplotlib.pyplot as plt
    grid_img = torchvision.utils.make_grid(w)
    plt.imshow(grid_img.permute(1,2,0).cpu())
    plt.show()



parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--project', type=str, default="APR")
parser.add_argument('--entity', type=str, default="hashmatshadab")
parser.add_argument('--wandb_mode', type=str, default="disabled")
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--end_epoch', type=int, default=35)
parser.add_argument('--save_epoch', type=int, default=1)
parser.add_argument('--resume', type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument('--chk_pth', type=str, default='....')
parser.add_argument('--batch_size', type=int, default=64, help='number of all reference images')
parser.add_argument('--mode', type=str, default='rotate', choices=["rotate","jigsaw"])
parser.add_argument('--data_dir', type=str, default='./data/ILSVRC2012_img_val')
parser.add_argument('--save_dir', type=str, default='./single_trained_models')
parser.add_argument('--fgsm_step', type=int, default=2)
parser.add_argument('--adv_train', type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument('--lr', type=float, default=0.001)


def create_json(args):
    """
     To create json file to save the arguments (args)
    """

    with open(f"{args.save_dir}/config_train_single.json", "w") as write_file:
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





if __name__ =="__main__":

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



    transform =  T.Compose([
            T.Resize((256,256)),
            T.CenterCrop(224),
            T.ToTensor()
        ])


    os.makedirs(args.save_dir + '/models', exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset = torchvision.datasets.folder.ImageFolder(root=args.data_dir, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = initialize_model(1)
    if args.resume:
        state = torch.load(f"{args.chk_pth}")
        model.load_state_dict(state)

    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.adv_train:
         train_unsup_adv(model, data_loader, optimizer, args)
    else:
        train_unsup(model, data_loader, optimizer, args)






