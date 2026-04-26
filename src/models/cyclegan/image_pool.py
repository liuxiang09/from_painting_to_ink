import random

import torch


class ImagePool:
    def __init__(self, pool_size: int):
        self.pool_size = pool_size
        self.num_imgs = 0
        self.images: list[torch.Tensor] = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        if self.pool_size <= 0:
            return images

        return_images = []
        for image in images.detach():
            image = image.unsqueeze(0)
            if self.num_imgs < self.pool_size:
                self.num_imgs += 1
                self.images.append(image)
                return_images.append(image)
            else:
                if random.random() > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)

        return torch.cat(return_images, dim=0)
