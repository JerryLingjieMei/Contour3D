import torch
import numpy as np


def combine_patch(visuals, image_paths):
    new_visuals = dict()
    for key, item in visuals.items():
        new_items = []
        items = torch.split(item, 4)
        for new_item in items:
            base = torch.zeros((new_item.shape[1], 512, 512)).to(new_item.device)
            for cycle in range(4):
                i = cycle // 2 * 128
                j = cycle % 2 * 128
                base[:, i:i + 384, j:j + 384] += new_item[cycle, :, :, :]
            base[:, 128:384, :] /= 2
            base[:, :, 128:384] /= 2
            new_items.append(base)
        new_visuals[key] = torch.stack(new_items)
    new_image_paths = [image_paths[i * 4] for i in range(len(image_paths) // 4)]
    return new_visuals, new_image_paths
