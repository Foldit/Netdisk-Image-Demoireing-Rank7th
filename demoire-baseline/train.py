import argparse
import os.path
import random
import time
import datetime
import sys

import numpy as np
import paddle
from PIL import Image

from transforms import RandomHorizontalFlip, Resize, Normalize,Rotate,Crop
from dataset import Dataset
from model import WaveletTransform, WDNet
from losses import LossNetwork
from visualdl import LogWriter

logWriter = LogWriter('/home/aistudio/demoire-baseline/log')

def parse_args():
    parser = argparse.ArgumentParser(description='Model training')

    parser.add_argument(
        '--dataset_root',
        dest='dataset_root',
        help='The path of dataset root',
        type=str,
        default=None)

    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='batch_size',
        type=int,
        default=8
    )

    parser.add_argument(
        '--max_epochs',
        dest='max_epochs',
        help='max_epochs',
        type=int,
        default=100
    )

    parser.add_argument(
        '--log_iters',
        dest='log_iters',
        help='log_iters',
        type=int,
        default=10
    )

    parser.add_argument(
        '--save_interval',
        dest='save_interval',
        help='save_interval',
        type=int,
        default=10
    )

    parser.add_argument(
        '--sample_interval',
        dest='sample_interval',
        help='sample_interval',
        type=int,
        default=100
    )

    parser.add_argument(
        '--seed',
        dest='seed',
        help='random seed',
        type=int,
        default=1234
    )

    return parser.parse_args()


def sample_images(epoch, i, real_A, real_B, fake_B):
    data, pred, label = real_A * 255, fake_B * 255, real_B * 255
    pred = paddle.clip(pred.detach(), 0, 255)

    data = data.cast('int64')
    pred = pred.cast('int64')
    label = label.cast('int64')
    h, w = pred.shape[-2], pred.shape[-1]
    img = np.zeros((h, 1 * 3 * w, 3))
    for idx in range(0, 1):
        row = idx * h
        tmplist = [data[idx], pred[idx], label[idx]]
        for k in range(3):
            col = k * w
            tmp = np.transpose(tmplist[k], (1, 2, 0))
            img[row:row + h, col:col + w] = np.array(tmp)
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    if not os.path.exists("./train_result"):
        os.makedirs("./train_result")
    img.save("./train_result/%03d_%06d.png" % (epoch, i))


def loss_textures(x, y, nc=3, alpha=1.2, margin=0):
    xi = x.reshape([x.shape[0], -1, nc, x.shape[2], x.shape[3]])
    yi = y.reshape([y.shape[0], -1, nc, y.shape[2], y.shape[3]])

    xi2 = paddle.sum(xi * xi, axis=2)
    yi2 = paddle.sum(yi * yi, axis=2)
    out = paddle.nn.functional.relu(yi2 * alpha - xi2 + margin)

    return paddle.mean(out)


def compute_l1_loss(input, output):
    return paddle.mean(paddle.abs(input - output))


def main(args):
    paddle.seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    transforms = [
        RandomHorizontalFlip(),
        Rotate(),
        Crop(target=(512,512)),
        Resize(target_size=(512, 512)),
        Normalize()
    ]
    dataset = Dataset(dataset_root=args.dataset_root, transforms=transforms)
    dataloader = paddle.io.DataLoader(dataset, batch_size=args.batch_size,
                                      num_workers=0, shuffle=True, drop_last=True,
                                      return_list=True)

    # Loss functions
    criterion_pixelwise = paddle.nn.L1Loss()  # smoothl1loss()
    lossnet = LossNetwork(pretrained="./vgg.pdparams")

    wavelet_dec = WaveletTransform(scale=2, dec=True)
    wavelet_rec = WaveletTransform(scale=2, dec=False)

    generator = WDNet()
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=3e-4, T_max=300000)
    # scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=3e-4, milestones=[100000,200000,250000], gamma=0.1)
    optimizer = paddle.optimizer.AdamW(parameters=generator.parameters(), learning_rate=scheduler,
                                      beta1=0.9, beta2=0.999,weight_decay=1e-4)

    prev_time = time.time()
    for epoch in range(1, args.max_epochs + 1):
        for i, data_batch in enumerate(dataloader):
            real_A = data_batch[0]
            real_B = data_batch[1]

            target_wavelets = wavelet_dec(real_B)
            wavelets_lr_b = target_wavelets[:, 0:3, :, :]
            wavelets_sr_b = target_wavelets[:, 3:, :, :]

            source_wavelets = wavelet_dec(real_A)

            wavelets_fake_B_re = generator(source_wavelets)

            fake_B = wavelet_rec(wavelets_fake_B_re) + real_A

            wavelets_fake_B = wavelet_dec(fake_B)
            wavelets_lr_fake_B = wavelets_fake_B[:, 0:3, :, :]
            wavelets_sr_fake_B = wavelets_fake_B[:, 3:, :, :]

            loss_GAN = 0.0

            loss_pixel = criterion_pixelwise(fake_B, real_B)  # .................................
            tensor_c = paddle.to_tensor(
                np.array([123.6800, 116.7790, 103.9390]).astype(np.float32).reshape((1, 3, 1, 1)))

            # preceptual loss
            loss_fake_B = lossnet(fake_B * 255 - tensor_c)
            loss_real_B = lossnet(real_B * 255 - tensor_c)
            p0 = compute_l1_loss(fake_B * 255 - tensor_c, real_B * 255 - tensor_c) * 2
            p1 = compute_l1_loss(loss_fake_B['relu1'], loss_real_B['relu1']) / 2.6
            p2 = compute_l1_loss(loss_fake_B['relu2'], loss_real_B['relu2']) / 4.8

            loss_p = p0 + p1 + p2  # +p3+p4+p5

            loss_lr = compute_l1_loss(wavelets_lr_fake_B[:, 0:3, :, :], wavelets_lr_b)
            loss_sr = compute_l1_loss(wavelets_sr_fake_B, wavelets_sr_b)
            loss_t = loss_textures(wavelets_sr_fake_B, wavelets_sr_b)

            loss_G = 0.001 * loss_GAN + (
                        1.1 * loss_p) + loss_sr * 100 + loss_lr * 10 + loss_t * 5  # +  loss_tv  loss_pixel

            loss_G.backward()

            optimizer.step()
            generator.clear_gradients()

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = args.max_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            scheduler.step()
            logWriter.add_scalar(tag='loss_Generator',step=epoch*len(dataloader)+i,value=loss_G.numpy()[0])
            logWriter.add_scalar(tag='preceptual loss',step=epoch*len(dataloader)+i,value=loss_p.numpy()[0])
            logWriter.add_scalar(tag='loss_sr',step=epoch*len(dataloader)+i,value=loss_sr.numpy()[0])
            logWriter.add_scalar(tag='loss_textures',step=epoch*len(dataloader)+i,value=loss_t.numpy()[0])
            logWriter.add_scalar(tag='loss_pixel',step=epoch*len(dataloader)+i,value=loss_pixel.numpy()[0])  
            logWriter.add_scalar(tag='lr',step=epoch*len(dataloader)+i,value=optimizer.get_lr())  
            if i % args.log_iters == 0:
                sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [G loss: %f, pixel: %f] ETA: %s" %
                                 (epoch, args.max_epochs,
                                  i, len(dataloader),
                                  loss_G.numpy()[0],
                                  loss_pixel.numpy()[0],
                                  time_left))

            if i % args.sample_interval == 0:
                sample_images(epoch, i, real_A, real_B, fake_B)

        if epoch % args.save_interval == 0:
            current_save_dir = os.path.join("train_result", "model", f'epoch_{epoch}')
            if not os.path.exists(current_save_dir):
                os.makedirs(current_save_dir)
            paddle.save(generator.state_dict(),
                        os.path.join(current_save_dir, 'model.pdparams'))
            paddle.save(optimizer.state_dict(),
                        os.path.join(current_save_dir, 'model.pdopt'))


if __name__ == '__main__':
    args = parse_args()
    main(args)
