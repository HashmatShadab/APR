import os
import torchvision.transforms as T
import torchvision
import argparse
import numpy as np
from torch.backends import cudnn
from model import *
from our_dataset import OUR_dataset
from utils import *
from classification import classify
import scipy.stats as st
import json
import wandb
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Attack')
parser.add_argument('--project', type=str, default="APR")
parser.add_argument('--entity', type=str, default="hashmatshadab")
parser.add_argument('--wandb_mode', type=str, default="disabled")
parser.add_argument('--chk_pth', type=str, default="trained_models/models/0.pth")
parser.add_argument('--epsilon', type=float, default=0.1)
parser.add_argument('--ila_niters', type=int, default=100)
parser.add_argument('--ce_niters', type=int, default=200)
parser.add_argument('--ce_epsilon', type=float, default=0.1)
parser.add_argument('--ce_alpha', type=float, default=1.0)
parser.add_argument('--n_imgs', type=int, default=20)
parser.add_argument('--save_dir', type=str, default='./adv_images/test')
parser.add_argument('--ce_method', type=str, default='ifgsm')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=2500)
parser.add_argument('--save_results', type=str, default='results', help="name of file for saving classification scores"
                                                                        "on various models")
parser.add_argument('--batch_size', type=int, default=64, help="batch size for the dataloader used for evaluation "
                                                               "classification accuracy on various models")


args = parser.parse_args()



wandb.init(project=args.project, entity=args.entity, mode=args.wandb_mode, name=args.save_dir.split("/")[-1])


class ILA(torch.nn.Module):
    def __init__(self):
        super(ILA, self).__init__()
    def forward(self, ori_mid, tar_mid, att_mid):
        bs = ori_mid.shape[0]
        ori_mid = ori_mid.view(bs, -1)
        tar_mid = tar_mid.view(bs, -1)
        att_mid = att_mid.view(bs, -1)
        W = att_mid - ori_mid
        V = tar_mid - ori_mid
        V = V / V.norm(p=2,dim=1, keepdim=True)
        ILA = (W*V).sum() / bs
        return ILA



def save_attack_img(img, file_dir):
    T.ToPILImage()(img.data.cpu()).save(file_dir)

def initialize_model(decoder_num):
    model = autoencoder(input_nc=3, output_nc=3, n_blocks=3, decoder_num=decoder_num)
    model = nn.Sequential(
        Normalize(),
        model,
    )
    model.to(device)
    return model

##define TI


def attack_ila(model, ori_img, tar_img, attack_niters, eps):
    """

    :param model:
    :param ori_img:
    :param tar_img:
    :param attack_niters:
    :param eps:
    :return:
    """
    # targ_img is the attacked img
    model.eval()
    ori_img = ori_img.to(device)
    img = ori_img.clone()
    with torch.no_grad():
        # get output of the encoder for tar_img and ori_img without computing gradients
        _, tar_h_feats,_ = model(tar_img)
        _, ori_h_feats,_ = model(ori_img)
        # ori_h_feats are the features from the original image
        # tar_h_feats are the features from the attacked images before ila
    for i in range(attack_niters):
        img.requires_grad_(True)
        _, att_h_feats,_ = model(img)
        # att_h_feats are features computed after after the orig image in the loop
        loss = ILA()(ori_h_feats.detach(), tar_h_feats.detach(), att_h_feats)
        if (i+1) % 50 == 0:
            print('\r ila attacking {}, {:0.4f}'.format(i+1, loss.item()),end=' ')
        loss.backward()
        input_grad = img.grad.data.sign()
        img = img.data + 1. / 255 * input_grad
        img = torch.where(img > ori_img + eps, ori_img + eps, img)
        img = torch.where(img < ori_img - eps, ori_img - eps, img)
        img = torch.clamp(img, min=0, max=1)
    print('')
    return img.data



def attack_ce_unsup(model, ori_img,args, attack_niters, eps,n_imgs, ce_method, attack_loss, iter):
    """
    For baseline gradient attack (ce_method).
    Applied on models trained using rotate/jigsaw/masking approach.
    :param model:
    :param ori_img: The original input image
    :param attack_niters: Number of  baseline-gradient attack iterations
    :param eps: Maximum perturbation rate for baseline attack
    :param alpha: Scaling parameter for adversarial loss
    :param n_imgs: Number of images
    :param ce_method: The gradient- based baseline attack used (IFGSM or PGD)
    :return: Returns the image with adversarial loss maximized within the bound.
    """
    model.eval()
    ori_img = ori_img.to(device)
    nChannels = 3
    tar_img = []
    for i in range(2 * n_imgs):
        tar_img.append(ori_img[i].unsqueeze(0))

    tar_img = torch.cat(tar_img, dim=0)

    img = ori_img.clone()
    attack_loss[iter] = []
    for i in range(attack_niters):
        if ce_method == 'ifgsm':
            img_x = img
        # In our implementation of PGD, we incorporate randomness at each iteration to further enhance the transferability
        elif ce_method == 'pgd':
            img_x = img + img.new(img.size()).uniform_(-eps, eps)

        img_x.requires_grad_(True)
        outs, _,_ = model(img_x)

        outs = outs[0]

        loss = nn.MSELoss(reduction='none')(outs, tar_img).sum() / (2*n_imgs*nChannels * 224 * 224)


        attack_loss[iter].append(loss.item())

        if (i + 1) % 50 == 0 or i == 0:
            print('\r attacking {}, {:0.4f}'.format(i, loss.item()), end=' ')
        loss.backward()
        input_grad = img_x.grad.data.sign()
        img = img.data + 1. / 255 * input_grad
        img = torch.where(img > ori_img + eps, ori_img + eps, img)
        img = torch.where(img < ori_img - eps, ori_img - eps, img)
        img = torch.clamp(img, min=0, max=1)
    print('')
    return img.data



def plot_grid(w):
    import matplotlib.pyplot as plt
    grid_img = torchvision.utils.make_grid(w)
    plt.imshow(grid_img.permute(1,2,0).cpu())
    plt.show()

def create_json(args):

    with open(f"{args.save_dir}/config_single_model_attack.json", "w") as write_file:
        json.dump(args.__dict__, write_file, indent=4)



if __name__ == '__main__':
    """
    Attacking a single autoencoder trained on a large dataset, by defaulat in an unsupervised fashion. The attack 
    also is fully unsupervised,  MSE(Adv, Org) objective function in this case.
    """
    SEED = 0
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    print(args)
    config = wandb.config
    config.update(args)
    save_dir = args.save_dir
    n_imgs = args.n_imgs // 2

    n_decoders = 1

    os.makedirs(save_dir, exist_ok=True)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    batch_size = n_imgs * 2
    epsilon = args.epsilon
    ce_epsilon = args.ce_epsilon
    ila_niters = args.ila_niters
    ce_niters = args.ce_niters
    ce_alpha = args.ce_alpha
    ce_method = args.ce_method
    assert ce_method in ['ifgsm', 'pgd']
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')




    trans = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop(224),
        T.ToTensor()
    ])
    dataset = OUR_dataset(data_dir='data/ILSVRC2012_img_val',
                          data_csv_dir='data/selected_data.csv',
                          mode='attack',
                          img_num=n_imgs,
                          transform=trans)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = False, num_workers = 1)
    create_json(args)
    fig, ax = plt.subplots()


    model = initialize_model(decoder_num=n_decoders)
    model.load_state_dict(torch.load(args.chk_pth))

    model.to(device)
    model.eval()

    for data_ind, (ori_img, _) in enumerate(dataloader):
        if not args.start <= data_ind < args.end:
            continue


        ori_img = ori_img.to(device)
        attack_loss = {}

        old_att_img = attack_ce_unsup(model, ori_img,args, attack_niters=ce_niters,
                                                      eps=ce_epsilon, n_imgs=n_imgs,
                                                      ce_method=ce_method,
                                                      attack_loss=attack_loss, iter=data_ind)
        xs = [x for x in range(len(attack_loss[data_ind]))]
        ax.plot(xs, attack_loss[data_ind], label=f"Model_{data_ind}")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Loss")
        ax.set_title(f"Model_{args.model}_{data_ind}")
        wandb.log({f'plot': ax})

        att_img = attack_ila(model, ori_img, old_att_img, ila_niters, eps=epsilon)
        for save_ind in range(batch_size):
            file_path, file_name = dataset.imgs[data_ind * 2*n_imgs + save_ind][0].split('/')[-2:]
            os.makedirs(save_dir + '/' + file_path, exist_ok=True)
            save_attack_img(img=att_img[save_ind],
                            file_dir=os.path.join(save_dir, file_path, file_name[:-5]) + '.png')
            print('\r', data_ind * batch_size + save_ind, 'images saved.', end=' ')

    classify(save_dir=args.save_dir, batch_size=args.batch_size, save_results=args.save_results)