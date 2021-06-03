import argparse
import os
from PIL import Image
import glob


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

    cropped_cnt = 0
    total_w = 0
    total_h = 0
    for i, image_file in enumerate(image_files):
        if i % 10 == 0:
            print(i, len(image_files), image_file)
        im = Image.open(image_file)
        bbox = im.getbbox()
        im = im.crop(bbox)
        w, h = im.size
        if w == 1280 and h == 720:
            continue
        total_w += w
        total_h += h
        cropped_cnt += 1
        im.save(os.path.join(args.output_dir, os.path.splitext(os.path.basename(image_file))[0] + ".jpg"))
    print("avg width", float(total_w) / cropped_cnt, "avg height", float(total_h) / cropped_cnt)
    print("done")


if __name__ == '__main__':
    main()
