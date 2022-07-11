import torch
import torch.nn as nn

from torchattacks.attack import Attack


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



class FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.007)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FGSM(model, eps=0.007)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, eps=0.007, mode="rotate"):
        super().__init__("FGSM", model)
        self.mode = mode
        self.eps = eps
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, target_images):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        target_images = target_images.clone().detach().to(self.device)


        loss = nn.MSELoss()

        images.requires_grad = True
        outputs, _, _ = self.model(images)

        # Calculate loss
        if self.mode == "rotate":
            cost = loss(outputs[0], target_images)
        else:
            gen_img = torch.cat(outputs, dim=0)
            cost = loss(gen_img, target_images)

        # Update adversarial images
        grad = torch.autograd.grad(cost, images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images = images + self.eps*grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images
