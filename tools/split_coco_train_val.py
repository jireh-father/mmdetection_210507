import argparse

import random
import json
import os
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate metric of the '
                                                 'results saved in pkl format')
    parser.add_argument('--annotation_file', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--image_dir', type=str, default=None)
    parser.add_argument('--train_ratio', type=float, default=0.9)
    parser.add_argument('--random_seed', type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    random.seed(args.random_seed)

    anno = json.load(open(args.annotation_file))

    images = anno["images"]
    train_cnt = round(len(images) * args.train_ratio)
    train_images = images[:train_cnt]
    val_images = images[train_cnt:]

    print("total", len(images))
    print("train", len(train_images))
    print("val", len(val_images))

    train_images_dict = {im['id']: im for im in train_images}
    val_images_dict = {im['id']: im for im in val_images}

    train_anno = []
    val_anno = []
    for tmp_anno in anno["annotations"]:
        if tmp_anno["image_id"] in train_images_dict:
            train_anno.append(tmp_anno)
        elif tmp_anno["image_id"] in val_images_dict:
            val_anno.append(tmp_anno)
        else:
            raise Exception("no image id in anno", tmp_anno)

    for i, image_item in enumerate(train_images):
        print(i, len(train_images))
        image_path = os.path.join(args.image_dir, image_item["file_name"])
        im = cv2.imread(image_path)
        height, width, _ = im.shape
        image_item['height'] = height
        image_item['width'] = width

    for i, image_item in enumerate(val_images):
        print(i, len(val_images))
        image_path = os.path.join(args.image_dir, image_item["file_name"])
        im = cv2.imread(image_path)
        height, width, _ = im.shape
        image_item['height'] = height
        image_item['width'] = width

    train_dict = {
        "info": anno["info"],
        "licenses": anno["info"],
        "categories": [{'id': 1, 'name': 'plant', 'supercategory': 'plant'}],
        "images": train_images,
        "annotations": train_anno
    }

    val_dict = {
        "info": anno["info"],
        "licenses": anno["info"],
        "categories": [{'id': 1, 'name': 'plant', 'supercategory': 'plant'}],
        "images": val_images,
        "annotations": val_anno
    }



    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(train_dict, open(os.path.join(args.output_dir, "train.json"), "w+"))
    json.dump(val_dict, open(os.path.join(args.output_dir, "val.json"), "w+"))

    print("done")


if __name__ == '__main__':
    main()
