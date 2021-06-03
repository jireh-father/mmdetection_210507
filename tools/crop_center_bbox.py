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

    for i, image_file in enumerate(image_files):
        if i % 10 == 0:
            print(i, len(image_files), image_file)
        im = Image.open(image_file)
        bbox = im.getbbox()
        im = im.crop(bbox)
        im.save(os.path.join(args.output_dir, os.path.splitext(os.path.basename(image_file))[0] + ".jpg"))
    print("done")


if __name__ == '__main__':
    main()
