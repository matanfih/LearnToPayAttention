import PIL
import numpy as np
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as utils
import torchvision.transforms as transforms


def _worker_init_fn_(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


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


def visual_test_image_softmax(model, image, transform_test, min_up_factor=1, outpath='/tmp'):
    print("test image: %s, outpath: %s" % (image, outpath))

    #p2t = transforms.PILToTensor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = transform_test(PIL.Image.open(image).convert('L'))
    img = img.float()

    inputs = img

    inputs = inputs.unsqueeze(1)
    inputs = inputs.to(device)

    pred, c1, c2, c3 = model(inputs)
    inputs = inputs[:, 0, :, :]

    attentions = [
        (c1, 1, os.path.join(outpath, 'attn1_' + os.path.basename(image))),
        (c2, 2, os.path.join(outpath, 'attn2_' + os.path.basename(image))),
        (c3, 3, os.path.join(outpath, 'attn3_' + os.path.basename(image)))
          ]

    utils.save_image(inputs, os.path.join(outpath, 'orig_' + os.path.basename(image)))

    if c1 is not None:
        attn1 = visualize_attn_softmax(inputs, c1, up_factor=min_up_factor, nrow=6)
        utils.save_image(attn1, os.path.join(outpath, 'attn1_' + os.path.basename(image)))
    if c2 is not None:
        attn2 = visualize_attn_softmax(inputs, c2, up_factor=min_up_factor * 2, nrow=6)
        utils.save_image(attn2, os.path.join(outpath, 'attn2_' + os.path.basename(image)))
    if c3 is not None:
        attn3 = visualize_attn_softmax(inputs, c3, up_factor=min_up_factor * 4, nrow=6)
        utils.save_image(attn3, os.path.join(outpath, 'attn3_' + os.path.basename(image)))

    return


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
    #pace = os.path.join(nih, "pacemakers/00028060_001.png")
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