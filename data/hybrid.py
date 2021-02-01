import cv2
import random
import numpy as np
import torch
from torchvision.transforms import GaussianBlur
from matplotlib import pyplot as plt
from itertools import permutations


def hybrid_image_opencv():
    kernel = (49, 49)
    image1 = cv2.cvtColor(cv2.imread('a0010-jmac_MG_4807.jpg'), cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(cv2.imread('a0014-WP_CRW_6320.jpg'), cv2.COLOR_BGR2RGB)

    image_low = cv2.GaussianBlur(image1, kernel, 0)
    image_high = image2 - cv2.GaussianBlur(image2, kernel, 0)

    image_hybrid = np.clip(image_high + image_low, 0, 255)

    plt.subplot(1, 3, 1)
    plt.imshow(image_low)
    plt.subplot(1, 3, 2)
    plt.imshow(np.clip(image_high + 127, 0, 255))
    plt.subplot(1, 3, 3)
    plt.imshow(image_hybrid)

    plt.show()


def hybrid_image_pytorch():
    """ Do not use: slow, output is grayscale """
    kernel = (49, 49)
    image1 = cv2.cvtColor(cv2.imread('a0010-jmac_MG_4807.jpg'), cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(cv2.imread('a0014-WP_CRW_6320.jpg'), cv2.COLOR_BGR2RGB)
    image1, image2 = torch.tensor(np.transpose(image1, (2, 0, 1))), torch.tensor(np.transpose(image2, (2, 0, 1)))

    image_low = GaussianBlur(kernel)(image1).numpy()
    image_high = image2 - GaussianBlur(kernel)(image2).numpy()

    image_low, image_high = np.transpose(image_low, (1, 2, 0)), np.transpose(image_high, (1, 2, 0))
    image_hybrid = np.clip(image_high + image_low, 0, 255)

    plt.subplot(1, 3, 1)
    plt.imshow(image_low)
    plt.subplot(1, 3, 2)
    plt.imshow(np.clip(image_high + 127, 0, 255))
    plt.subplot(1, 3, 3)
    plt.imshow(image_hybrid)
    plt.show()
    pass


def compose_hybrid_image(src_low, src_high, kernel=(9, 9)):
    """
    Compose a hybrid image based on a pair of inputs specifying the low and high frequency components
    :param src_low: source image that contains the low frequency component, shape (H x W x 3)
    :param src_high: source image that contains the high frequency component, shape (H x W x 3)
    :param kernel: kernel size of Gaussian blur
    :return: hybrid image, shape (H x W x 3)
    """
    image_low = cv2.GaussianBlur(src_low, kernel, 0)
    image_high = src_high - cv2.GaussianBlur(src_high, kernel, 0)
    return image_low + image_high


def get_hybrid_images(image_batch, kernel=(9, 9), gt_batch=None, num_classes=10):
    """
    Generate hybrid images for an image batch
    :param num_classes:
    :param image_batch:
    :param kernel:
    :param gt_batch:
    :return:
    """
    if len(image_batch) < 2:
        return None

    images = image_batch.permute(0, 2, 3, 1).numpy() if torch.is_tensor(image_batch) else image_batch
    hybrid_images = []
    indices_pairs = random.choices(list(permutations(range(len(images)), 2)), k=len(images))
    for indices_pair in indices_pairs:
        hybrid_images.append(compose_hybrid_image(images[indices_pair[0]], images[indices_pair[1]], kernel))
    hybrid_images = np.stack(hybrid_images)
    hybrid_images = torch.tensor(hybrid_images).permute(0, 3, 1, 2) if torch.is_tensor(image_batch) else hybrid_images

    if gt_batch is not None:
        multi_labels = []
        for indices_pair in indices_pairs:
            l = to_onehot(gt_batch[indices_pair[0]], num_classes) + to_onehot(gt_batch[indices_pair[1]], num_classes)
            multi_labels.append(l.clip(0, 1))
        return hybrid_images, torch.Tensor(multi_labels).squeeze()

    return hybrid_images


def to_onehot(labels, n_categories, dtype=torch.float32):
    b = np.zeros((1, n_categories))
    b[np.arange(1), labels] = 1

    return b


if __name__ == '__main__':
    hybrid_image_opencv()
    pass
