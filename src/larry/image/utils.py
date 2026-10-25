import torch


def pad_images(images: list[torch.Tensor]) -> torch.Tensor:
    # (C, H, W) images or latents, zero-padded at the bottom right to the batch's largest channels, height and width
    channels, height, width = (max(image.shape[dim] for image in images) for dim in range(3))
    padded = torch.zeros(len(images), channels, height, width, dtype=images[0].dtype if images else torch.float32)
    for i, image in enumerate(images):
        padded[i, :image.shape[0], :image.shape[1], :image.shape[2]] = image

    return padded
