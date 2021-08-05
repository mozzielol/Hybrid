import cv2
import numpy as np
import torch


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


def get_hybrid_images(image_batch, origin=False, kernel=(9, 9)):
    images = image_batch.permute(0, 2, 3, 1).cpu().numpy() if torch.is_tensor(image_batch) else image_batch
    hybrid_images = []
    # indices_pairs = random.choices(list(permutations(range(len(images)))), k=len(images))
    indices_pairs = np.random.choice(np.arange(images.shape[0]), size=len(images))
    for i, j in enumerate(indices_pairs):
        if origin and np.random.random_sample() > 0.5:
            hybrid_images.append(images[i])
        else:
            if np.random.random_sample() > 0.5:
                hybrid_images.append(compose_hybrid_image(images[i], images[j], kernel))
            else:
                hybrid_images.append(compose_hybrid_image(images[j], images[i], kernel))
    hybrid_images = np.stack(hybrid_images)
    hybrid_images = torch.tensor(hybrid_images).permute(0, 3, 1, 2) if torch.is_tensor(
        image_batch) else hybrid_images

    return hybrid_images
