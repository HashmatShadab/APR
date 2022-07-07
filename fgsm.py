import torch
import torch.nn as nn

from torchattacks.attack import Attack


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