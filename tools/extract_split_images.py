import argparse

import random
import json
import os
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate metric of the '
                                                 'results saved in pkl format')
    parser.add_argument('--annotation_file', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--image_dir', type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    anno = json.load(open(args.annotation_file))
    images = anno["images"]

    os.makedirs(args.output_dir, exist_ok=True)
    for i, im in enumerate(images):
        print(i, len(images))
        file_path = os.path.join(args.image_dir, im['file_name'])
        shutil.copy(file_path, args.output_dir)
    print("done")


if __name__ == '__main__':
    main()
