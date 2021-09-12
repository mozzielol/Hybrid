import numpy as np
import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F


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

def compose_hybrid_image(gaussian_blur, src_low, src_high, kernel=(9, 9)):
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


def get_hybrid_images(gaussian_blur, image_batch, kernel=(9, 9)):
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
    hybrid_images = compose_hybrid_image(gaussian_blur, images, images[idxs])
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
