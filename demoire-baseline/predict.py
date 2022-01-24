import argparse
import glob
import os.path

import paddle
import paddle.nn as nn
import cv2

from model import WDNet, WaveletTransform
from utils import load_pretrained_model


def parse_args():
    parser = argparse.ArgumentParser(description='Model testing')

    parser.add_argument(
        '--dataset_root',
        dest='dataset_root',
        help='The path of dataset root',
        type=str,
        default='/home/aistudio/data/moire_testB_dataset/')

    parser.add_argument(
        '--pretrained',
        dest='pretrained',
        help='The pretrained of model',
        type=str,
        default=None)

    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='batch_size',
        type=int,
        default=32
    )

    return parser.parse_args()

def main(args):
    model = WDNet()

    if args.pretrained is not None:
        load_pretrained_model(model, args.pretrained)

    wavelet_dec = WaveletTransform(scale=2, dec=True)
    wavelet_rec = WaveletTransform(scale=2, dec=False)
    # print(args.dataset_root)
    im_files = glob.glob(os.path.join(args.dataset_root, "*.jpg"))
    #print("怎么啥都没有")
    print(im_files)

    for i, im in enumerate(im_files):
        img = cv2.imread(im)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = paddle.to_tensor(img)
        img /= 255.0
        img = paddle.transpose(img, [2, 0, 1])
        model.eval()

        img = img.unsqueeze(0)
        img_in = wavelet_dec(img)
        img_out = model(img_in)
        img_out = wavelet_rec(img_out)
        img_out = nn.functional.interpolate(img_out, size=img.shape[2:], mode="bilinear")
        img_out = img_out + img
        img_out = img_out.squeeze(0)

        img_out = img_out * 255.0
        img_out = paddle.clip(img_out, 0, 255)
        img_out = paddle.transpose(img_out, [1, 2, 0])
        img_out = img_out.numpy()

        save_path = "output/pre"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_path, im.split('/')[-1]), img_out)


if __name__ == '__main__':
    args = parse_args()
    main(args)

