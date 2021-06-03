import argparse
import os
from PIL import Image
import glob
import numpy as np
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate metric of the '
                                                 'results saved in pkl format')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--image_dir', type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    image_files = glob.glob(os.path.join(args.image_dir, "*"))

    total_w = []
    total_h = []
    total_time = 0.
    for i, image_file in enumerate(image_files):
        if i % 10 == 0:
            print(i, len(image_files), image_file)
        start = time.time()
        im = Image.open(image_file)
        bbox = im.getbbox()
        im = im.crop(bbox)
        exec_time = time.time() - start
        w, h = im.size
        if w == 1280 and h == 720:
            continue
        total_time += exec_time
        total_w.append(w)
        total_h.append(h)
        im.save(os.path.join(args.output_dir, os.path.splitext(os.path.basename(image_file))[0] + ".jpg"))
    total_h = np.array(total_h)
    total_w = np.array(total_w)
    print("total_w.mean(), total_w.max(), total_w.min(), total_w.var(), total_w.std()")
    print(total_w.mean(), total_w.max(), total_w.min(), total_w.var(), total_w.std())
    print("total_h.mean(), total_h.max(), total_h.min(), total_h.var(), total_h.std()")
    print(total_h.mean(), total_h.max(), total_h.min(), total_h.var(), total_h.std())
    print("cropped cnt", len(total_h))
    print("avg time", total_time / len(total_h))
    print("done")


if __name__ == '__main__':
    main()
