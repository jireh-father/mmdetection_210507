import argparse

import random
import json
import os
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate metric of the '
                                                 'results saved in pkl format')
    parser.add_argument('--annotation_file', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--image_dir', type=str, default=None)
    parser.add_argument('--random_seed', type=int, default=1)
    parser.add_argument('--num_images', type=int, default=None)
    parser.add_argument('--check_image_size', action="store_true", default=False)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    random.seed(args.random_seed)

    anno = json.load(open(args.annotation_file))
    images = anno["images"]
    assert len(images) > args.num_images
    random.shuffle(images)
    images = images[:args.num_images]

    print("total", len(images))

    images_dict = {im['id']: im for im in images}

    new_anno = []
    for tmp_anno in anno["annotations"]:
        if tmp_anno["image_id"] in images_dict:
            new_anno.append(tmp_anno)
        else:
            raise Exception("no image id in anno", tmp_anno)

    if args.check_image_size:
        for i, image_item in enumerate(images):
            print(i, len(images))
            image_path = os.path.join(args.image_dir, image_item["file_name"])
            im = cv2.imread(image_path)
            height, width, _ = im.shape
            image_item['height'] = height
            image_item['width'] = width

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    anno['images'] = images
    anno['annotations'] = new_anno
    json.dump(anno, open(args.output_path, "w+"))

    print("done")


if __name__ == '__main__':
    main()
