import argparse
import json

import PIL
import torchvision.datasets.folder
from PIL import Image
from PIL import ImageFile
from torchvision import transforms as T
import os
import wandb
from model import *
from train_methods_single import train_unsup, train_unsup_adv
import numpy as np
PIL.Image.MAX_IMAGE_PIXELS = 933120000

from torch.backends import cudnn
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils import*

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
args = parser.parse_args()
wandb.init(project=args.project, entity=args.entity, mode=args.wandb_mode, name=args.save_dir.split("/")[-1])

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







if __name__ =="__main__":
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






