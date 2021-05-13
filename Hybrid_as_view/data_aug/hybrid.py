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


def generate_pairs_with_hybrid_images(seed_images, kernel=(9, 9), weights=(0.5, 0.5)):
    """
    Generate a batch of mixture of hybrid and original images, as well as the corresponding similarity matrix
    :param seed_images: a list of original seed images (N x D)
    :param kernel: kernel size for generating hybrid images
    :param weights: weights of hybrid and original images in the generated image list
    :return: a list of generated images (2N x D), as well as the similarity matrix (2N x 2N)
    """
    weights = (abs(w) / sum(weights) for w in weights)
    num_seed = len(seed_images)
    num_pure = round(2 * num_seed * weights[0])
    num_hybrid = 2 * num_seed - num_pure

    # Compute list of indices, each element corresponds to the seed index/indices of a output image
    pure_indices = [(idx,) for idx in np.random.choice(num_seed, num_pure, replace=True)]
    hybrid_indices = [tuple(np.random.choice(num_seed, 2, replace=False)) for i in range(num_hybrid)]
    composition_indices = pure_indices + hybrid_indices
    np.random.shuffle(composition_indices)

    # Generate output images
    generated_images = []
    for indices in composition_indices:
        if len(indices) > 1:
            hybrid_image = compose_hybrid_image(seed_images[indices[0]], seed_images[indices[1]], kernel=kernel)
            generated_images.append(hybrid_image)
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
    for i in range(num_seed):
        for j in range(i + 1):
            similarity_matrix[i, j] = measure_similarity(composition_indices[i], composition_indices[j])
            similarity_matrix[j, i] = measure_similarity(composition_indices[i], composition_indices[j])

    return generated_images, similarity_matrix
