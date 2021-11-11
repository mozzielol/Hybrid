import numpy as np
import math
import numbers
import torch
import torchgeometry as tgm
from torch import nn
from torch.nn import functional as F
from copy import deepcopy


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


def compose_hybrid_image(src_low, src_high, kernel=(9, 9), sigma=(1.5, 1.5)):
    """
    Compose a hybrid image based on a pair of inputs specifying the low and high frequency components
    :param src_low: source image that contains the low frequency component, shape (N x H x W x 3)
    :param src_high: source image that contains the high frequency component, shape (N x H x W x 3)
    :param kernel: kernel size of Gaussian blur Tuple
    :param sigma: standard deviation of the kernel
    :return: hybrid image, shape (N x H x W x 3)
    """
    image_low = tgm.image.gaussian_blur(src_low, kernel_size=kernel, sigma=sigma)
    image_high = src_high - tgm.image.gaussian_blur(src_high, kernel_size=kernel, sigma=sigma)
    return (image_low + image_high).clamp(0, 1)


def images_to_tensors(hybrid_images):
    hybrid_images = np.stack(hybrid_images)
    try:
        hybrid_images = torch.tensor(hybrid_images).permute(0, 3, 1, 2)
    except RuntimeError:
        hybrid_images = torch.tensor(hybrid_images).permute(1, 0, 4, 2, 3)
    return hybrid_images


def get_hybrid_images(image_batch, kernel=(9, 9), sigma=(1.5, 1.5)):
    """
    :param image_batch: batch of input images
    :param kernel: kernel for Gaussian blurring to generate hybrid images
    :param sigma: standard deviation of Gaussian kernel
    :return:
    """
    images = image_batch
    negative_paris = []

    idxs = []
    for i in np.arange(len(image_batch)):
        image_indices = np.arange(len(image_batch))
        image_indices = np.delete(image_indices, i)
        j = np.random.choice(len(image_indices))
        idxs.append(j)
        negative_paris.append(np.ones(len(image_batch), dtype=bool))
        negative_paris[i][i] = False
        negative_paris[i][image_indices[j]] = False
    hybrid_images = compose_hybrid_image(images, images[idxs], kernel, sigma)
    second_component = images[idxs]
    negative_paris = np.array(negative_paris)

    return hybrid_images, second_component, negative_paris


def generate_pairs_with_hybrid_images(seed_images, kernel=(15, 15), weights=(0.5, 0.5)):
    """
    Generate a batch of mixture of hybrid and original images, as well as the corresponding similarity matrix
    :param seed_images: a list of original seed images (N x D)
    :param kernel: kernel size for generating hybrid images
    :param weights: weights of hybrid and original images in the generated image list
    :return: a list of generated images (2N x D), as well as the similarity matrix (2N x 2N)
    """
    seed_images = seed_images.detach().numpy()
    weights = [abs(w) / sum(weights) for w in weights]
    num_seed = len(seed_images)
    num_pure = round(2 * num_seed * weights[0])
    num_hybrid = 2 * num_seed - num_pure

    # Compute list of indices, each element corresponds to the seed index/indices of a output image
    pure_indices = [(idx,) for idx in np.random.choice(num_seed, num_pure, replace=False)]
    hybrid_indices = [tuple(np.random.choice(num_seed, 2, replace=False)) for i in range(num_hybrid)]
    composition_indices = pure_indices + hybrid_indices
    np.random.shuffle(composition_indices)

    # Generate output images
    generated_images = []
    for indices in composition_indices:
        if len(indices) > 1:
            hybrid_image = compose_hybrid_image(seed_images[indices[0]], seed_images[indices[1]], kernel=tuple(kernel))
            generated_images.append(hybrid_image.clip(0, 1))  # either clipping or renormalization
        else:
            generated_images.append(seed_images[indices[0]])

    # Compute similarity matrix
    def measure_similarity(indices1, indices2):
        if len(indices1) == len(indices2):
            if indices1 == indices2:
                return 1  # (A), (A)
            else:
                if len(indices1) == 1:
                    return 0  # (A), (B)
                else:
                    if indices1[0] == indices2[0] or indices1[1] == indices2[1]:
                        return 0.3  # (A, B), (A, C)
                    elif indices1[0] == indices2[1]:
                        return 0.8 if indices1[1] == indices2[0] else 0.2  # (A, B), (B, A) - .8; (A, B), (C, A) - .2
                    elif indices1[1] == indices2[0]:
                        return 0.2  # (A, B), (B, C)
                    else:
                        return 0  # (A, B), (C, D)
        else:
            _h_indices, _p_indices = (indices1, indices2) if len(indices1) > len(indices2) else (indices2, indices1)
            return 0.5 if _p_indices[0] in _h_indices else 0  # (A, B), A - 0.5; (A, B), C - 0

    similarity_matrix = np.zeros((2 * num_seed,) * 2)
    for i in range(len(composition_indices)):
        for j in range(i + 1):
            similarity_matrix[i, j] = measure_similarity(composition_indices[i], composition_indices[j])
            similarity_matrix[j, i] = measure_similarity(composition_indices[i], composition_indices[j])

    return torch.Tensor(generated_images), torch.Tensor(similarity_matrix)


def rand_bbox(size, lam):
    """
    Generate random bounding box for patch replacement
    Reference: https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py#L279
    :param size:
    :param lam:
    :return:
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def compose_cutmix_image(src_a, src_b, beta=0):
    """
    Generate a batch of mixed sample using CutMix augmentation
    :param src_a: a batch of source images, shape N x D x H x W
    :param src_b: another batch of source images, shape N x D x H x W
    :param beta: beta distribution coefficient that controls the ratio (\lambda) of the bounding box
    :return: a batch of CutMixed images: Cut src_b to mix src_a
    """
    assert src_a.shape == src_b.shape, 'Images for CutMix should have exact same shape.'
    if beta > 0:
        result = deepcopy(src_a)
        lam = np.random.beta(beta, beta)
        rand_index = torch.randperm(src_a.size()[0]).cuda()
        bbx1, bby1, bbx2, bby2 = rand_bbox(src_a.size(), lam)
        result[:, :, bbx1:bbx2, bby1:bby2] = src_b[rand_index, :, bbx1:bbx2, bby1:bby2]
        return result
    else:
        return src_a


def compose_mixup_image(src_a, src_b, ratio_offset=0):
    """
    Generate a batch of mixed sample using Mixup augmentation
    :param src_a: a batch of source images, shape N x D x H x W
    :param src_b: another batch of source images, shape N x D x H x W
    :param ratio_offset: offset that controls the ratio of mixing two components, range [0, 0.5]
    :return:  a batch of Mixed-up images
    """
    assert src_a.shape == src_b.shape, 'Images for Mixup should have exact same shape.'
    ratio_offset = max(min(ratio_offset, 0.5), 0)
    alphas = 0.5 + ratio_offset - 2 * ratio_offset * torch.rand(len(src_a))
    results = [alpha * image1 + (1 - alpha) * image2 for alpha, image1, image2 in zip(alphas, src_a, src_b)]
    return torch.clamp(torch.stack(results), 0, 1)

