from PIL import ImageFilter
import random
import torch
from torchvision import transforms

def norm_mean_std(size):
    if size == 32:  # CIFAR10, CIFAR100
        normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    elif size == 64:  # Tiny-ImageNet
        normalize = transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
    elif size == 96:  # STL10
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    else:  # ImageNet
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return normalize


class GaussianBlur(object):
    """Gaussian blur augmentation """

    def __init__(self, sigma=None):
        if sigma is None:
            sigma = [.1, 2.]
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_color_distortion(s=1.0):
    """
    Color jitter from SimCLR paper
    @param s: is the strength of color distortion.
    """

    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


class SimCLRTransform:
    """
    Transform defined in SimCLR
    https://arxiv.org/pdf/2002.05709.pdf
    """

    def __init__(self, size):
        normalize = norm_mean_std(size)
        if size > 28:  # ImageNet
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(size=(size, size)),
                    transforms.RandomHorizontalFlip(),
                    get_color_distortion(s=1.0),
                    # transforms.ToTensor(),
                    normalize  # comment this line out for MNIST datasets
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(size=(size, size)),
                    transforms.RandomHorizontalFlip(),
                    # get_color_distortion(s=1.0),
                    # transforms.ToTensor(),
                    # normalize
                ]
            )

    def __call__(self, x):
        return self.transform(x), self.transform(x)

def apply_ssl_transform(inputs):
    X, Y = [], []
    ssl_transform = SimCLRTransform(size=224)
    for input in inputs:
        x, y = ssl_transform(input)
        X.append(x)
        Y.append(y)
    X = torch.stack(X).to(inputs.device)
    Y = torch.stack(Y, dim=0).to(inputs.device)
    return X, Y