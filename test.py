import os
import sys

import PIL.Image
import numpy as np
import torch
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.autograd import Variable
from model import Finetunemodel
from tqdm import tqdm

from file import MkdirSimple
from multi_read_data import MemoryFriendlyLoader

parser = argparse.ArgumentParser("SCI")
parser.add_argument('--data_path', type=str, default='./data/medium',
                    help='location of the data corpus')
parser.add_argument('--save_path', type=str, default='./results/medium', help='location of the data corpus')
parser.add_argument('--model', type=str, default='./weights/medium.pt', help='location of the data corpus')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')

args = parser.parse_args()
save_path = args.save_path
os.makedirs(save_path, exist_ok=True)

TestDataset = MemoryFriendlyLoader(img_dir=args.data_path, task='test')

test_queue = torch.utils.data.DataLoader(
    TestDataset, batch_size=1,
    pin_memory=True, num_workers=0)


def save_images(input, tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))

    image_file = os.path.join(save_path, 'light', path)
    concat_image_file = os.path.join(save_path, 'concat', path)
    MkdirSimple(image_file)
    MkdirSimple(concat_image_file)

    im.save(image_file)

    input_numpy = input[0].cpu().float().numpy()
    input_numpy = (np.transpose(input_numpy, (1, 2, 0)))
    input_img = Image.fromarray(np.clip(input_numpy * 255.0, 0, 255.0).astype('uint8'))

    concat_img = np.vstack([input_img, im])
    PIL.Image.fromarray(concat_img).save(concat_image_file)


def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    root_len = len(args.data_path.rstrip('/'))

    model = Finetunemodel(args.model)
    model = model.cuda()

    model.eval()
    with torch.no_grad():
        for input, image_name in tqdm(test_queue):
            if input is None:
                continue
            input = Variable(input, volatile=True).cuda()
            i, r = model(input)
            u_path = image_name[0][root_len+1:]
            save_images(input, r, u_path)



if __name__ == '__main__':
    main()
