import cv2
import numpy as np
import torch
import math
from util.util import get_device

def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1).to(get_device())

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter

gaussian_blur = get_gaussian_kernel(kernel_size=9)

def compose_hybrid_image(src_low, src_high, kernel=(9, 9)):
    """
    Compose a hybrid image based on a pair of inputs specifying the low and high frequency components
    :param src_low: source image that contains the low frequency component, shape (H x W x 3)
    :param src_high: source image that contains the high frequency component, shape (H x W x 3)
    :param kernel: kernel size of Gaussian blur
    :return: hybrid image, shape (H x W x 3)
    """
    image_low_pad = torch.nn.functional.pad(src_low, (4, 4, 4, 4), mode='reflect')
    image_high_pad = torch.nn.functional.pad(src_high, (4, 4, 4, 4),mode='reflect')
    image_low = gaussian_blur(image_low_pad)
    image_high = gaussian_blur(image_high_pad)
    return image_low + image_high


def images_to_tensors(hybrid_images):
    hybrid_images = np.stack(hybrid_images)
    try:
        hybrid_images = torch.tensor(hybrid_images).permute(0, 3, 1, 2)
    except RuntimeError:
        hybrid_images = torch.tensor(hybrid_images).permute(1, 0, 4, 2, 3)
    return hybrid_images


def get_hybrid_images(image_batch, kernel=(9, 9)):
    """
    :param image_batch: input images
    :param kernel: kernel for hybrid images
    :param return_sec_component: additionally return the second component which is used for constructing hybrid images
    :param return_other: additionally return another images
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
    hybrid_images = compose_hybrid_image(images, images[idxs])
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
