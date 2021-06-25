import argparse

import random
import json
import os
import cv2
from PIL import Image
from pycocotools import mask


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate metric of the '
                                                 'results saved in pkl format')
    parser.add_argument('--annotation_file', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--image_dir', type=str, default=None)
    parser.add_argument('--shortest_size', type=int, default=800)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    output_image_dir = os.path.join(args.output_dir, "images")
    output_anno_path = os.path.join(args.output_dir, "annotations.json")
    os.makedirs(output_image_dir, exist_ok=True)
    anno = json.load(open(args.annotation_file))

    images = anno["images"]
    print("total", len(images))

    images_dict = {im['id']: im for im in images}
    image_anno_dict = {}

    for tmp_anno in anno["annotations"]:
        if tmp_anno["image_id"] in images_dict:
            image_id = tmp_anno["image_id"]
            if image_id not in image_anno_dict:
                image_anno_dict[image_id] = []
            image_anno_dict[image_id].append(tmp_anno)
        else:
            raise Exception("no image")

    new_images = []
    new_annos = []

    for image_id in images_dict:
        image_info = images_dict[image_id]
        tmp_annos = image_anno_dict[image_id]
        w = image_info['width']
        h = image_info['height']

        if min(w, h) < args.shortest_size:
            new_images.append(image_info)
            new_annos += tmp_annos
            continue

        short_size = min(w, h)
        ratio = args.shortest_size / short_size
        if w > h:
            target_w = w * ratio
            target_h = args.shortest_size
        else:
            target_w = args.shortest_size
            target_h = h * ratio
        image_path = os.path.join(args.image_dir, image_info['file_name'])
        im = Image.open(image_path).convert("RGB")
        im = im.resize((target_w, target_h), Image.ANTIALIAS)
        out_fname = os.path.splitext(image_info['file_name'])[0] + ".jpg"
        im.save(os.path.join(output_image_dir, out_fname), quality=100)
        image_info['width'] = target_w
        image_info['height'] = target_h
        image_info['file_name'] = out_fname

        new_images.append(image_info)

        for tmp_anno in tmp_annos:
            seg = tmp_anno['segmentation'][0]
            for i in range(len(seg)):
                seg[i] = seg[i] * ratio
            seg = [seg]

            bbox = tmp_anno['bbox']
            for i in range(len(bbox)):
                bbox[i] = bbox[i] * ratio

            rles = mask.frPyObjects(seg, target_h, target_w)
            rle = mask.merge(rles)
            area = int(mask.area(rle))

            new_anno = tmp_anno
            new_anno['segmentations'] = seg
            new_anno['bbox'] = bbox
            new_anno['area'] = area
            new_annos.append(new_anno)

    anno["images"] = new_images
    anno["annotations"] = new_annos
    json.dump(anno, open(output_anno_path, "w+"))

    print("done")


if __name__ == '__main__':
    main()
