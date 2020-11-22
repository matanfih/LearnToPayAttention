import numpy as np
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as utils
import torchvision.transforms as transforms

def visualize_attn_softmax(I, c, up_factor, nrow):
    # image
    img = I.permute((1,2,0)).cpu().numpy()
    # compute the heatmap
    N,C,W,H = c.size()
    a = F.softmax(c.view(N,C,-1), dim=2).view(N,C,W,H)
    if up_factor > 1:
        a = F.interpolate(a, scale_factor=up_factor, mode='bilinear', align_corners=False)
    attn = utils.make_grid(a, nrow=nrow, normalize=True, scale_each=True)
    attn = attn.permute((1,2,0)).mul(255).byte().cpu().numpy()
    attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
    attn = cv2.cvtColor(attn, cv2.COLOR_BGR2RGB)
    attn = np.float32(attn) / 255
    # add the heatmap to the image
    vis = 0.6 * img + 0.4 * attn
    return torch.from_numpy(vis).permute(2,0,1)

def visualize_attn_sigmoid(I, c, up_factor, nrow):
    # image
    img = I.permute((1,2,0)).cpu().numpy()
    # compute the heatmap
    a = torch.sigmoid(c)
    if up_factor > 1:
        a = F.interpolate(a, scale_factor=up_factor, mode='bilinear', align_corners=False)
    attn = utils.make_grid(a, nrow=nrow, normalize=False)
    attn = attn.permute((1,2,0)).mul(255).byte().cpu().numpy()
    attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
    attn = cv2.cvtColor(attn, cv2.COLOR_BGR2RGB)
    attn = np.float32(attn) / 255
    # add the heatmap to the image
    vis = 0.6 * img + 0.4 * attn
    return torch.from_numpy(vis).permute(2,0,1)


def per_img_local_center(fimg):
    # example of per-channel centering (subtract mean)
    from PIL import Image
    # load image
    image = Image.open(fimg)
    pixels = np.asarray(image)
    # convert from integers to floats
    pixels = pixels.astype('float32')
    # calculate per-channel means and standard deviations
    means = pixels.mean(axis=(0, 1), dtype='float64')
    print('Means: %s' % means)
    print('Mins: %s, Maxs: %s' % (pixels.min(axis=(0, 1)), pixels.max(axis=(0, 1))))
    # per-channel centering of pixels
    pixels -= means
    # confirm it had the desired effect
    means = pixels.mean(axis=(0, 1), dtype='float64')
    print('Means: %s' % means)
    print('Mins: %s, Maxs: %s' % (pixels.min(axis=(0, 1)), pixels.max(axis=(0, 1))))


def per_img_std(img, transform, L=True):
    # example of per-channel pixel standardization
    from PIL import Image
    # load image

    if L:
        image = Image.open(img).convert('L')
    else:
        image = Image.open(img)

    if transform is not None:
        image = transform(image)

    pixels = np.asarray(image)
    # convert from integers to floats
    pixels = pixels.astype('float32')
    # calculate per-channel means and standard deviations
    means = pixels.mean(axis=(0, 1), dtype='float64')
    stds = pixels.std(axis=(0, 1), dtype='float64')
    print('Means: %s, Stds: %s' % (mean(means), mean(stds)))
    # per-channel standardization of pixels
    #
    # _pixels = (pixels - means) / stds
    # # confirm it had the desired effect
    # _means = _pixels.mean(axis=(0, 1), dtype='float64')
    # _stds = _pixels.std(axis=(0, 1), dtype='float64')
    # print('Means: %s, Stds: %s' % (_means, _stds))
    return mean(means), mean(stds)


if __name__ == "__main__":
    from statistics import mean
    import glob
    root = "/data/matan/nih"
    #pace = os.path.join(root, "pacemakers/00028060_001.png")
    #per_img_local_center(pace)
    #print("std:")
    _transform = transforms.Compose([transforms.Resize(128), transforms.ToTensor()])

    images = glob.glob("{}/images_*/*.png".format(root))
    mean_std = [per_img_std(img, _transform) for img in images]
    means = [m[0] for m in mean_std]
    stds = [s[1] for s in mean_std]

    print("dataset mean: %s, std: %s" % (round(mean(means), 3), round(mean(stds), 3)))

    #per_img_std(pace, _transform)
    #per_img_std(pace, None)