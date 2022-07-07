import argparse
import json

import matplotlib.pyplot as plt
import torchvision.transforms as T
import wandb
from torch.backends import cudnn
from torch.utils.data import Dataset

from model import *
from our_dataset import OUR_dataset
from train_methods import train_unsup, train_prototypical, train_adv_prototypical, train_adv_unsup
from utils import *

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--project', type=str, default="No-Box Attack")
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
parser.add_argument('--fgsm_step', type=int, default=4)
parser.add_argument('--clip_eps', type=int, default=4)
parser.add_argument('--adv_train', type=lambda x: (str(x).lower() == 'true'), default=False)

args = parser.parse_args()
wandb.init(project=args.project, entity=args.entity, mode=args.wandb_mode, name=args.save_dir.split("/")[-1])

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


if __name__ == '__main__':

    """
        Training of multiple autoencoders using rotation, jigsaw, prototypical, unsup_naive and masking methods.
        Each autoencoder is trained on only 20 images (10 from each class) using the above methods.
        In total 250 autoencoders are trained on a subset 0f 5000 images (10 from each class) from ImageNetval.
    """



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

