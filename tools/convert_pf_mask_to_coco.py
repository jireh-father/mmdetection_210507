import argparse

import json
import os
import glob
import pandas as pd
import sys
from pycocotools import mask
import cv2
import numpy as np
from imantics import Polygons, Mask


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate metric of the '
                                                 'results saved in pkl format')
    parser.add_argument('--image_dir', type=str,
                        default='/media/irelin/data_disk/dataset/perfitt/2.Segmentation_original/first/origin')
    parser.add_argument('--mask_dir', type=str,
                        default='/media/irelin/data_disk/dataset/perfitt/2.Segmentation_original/first/seg')
    parser.add_argument('--output_path', type=str,
                        default='/media/irelin/data_disk/dataset/perfitt/2.Segmentation_original/first/coco.json')
    parser.add_argument('--wrong_label_file', type=str,
                        default='/home/irelin/source/perfitt_footerist/wrong_labels.txt')
    args = parser.parse_args()
    return args


def get_except_img_files(except_files):
    except_img_files = []
    for except_file in except_files:
        df = pd.read_csv(except_file)
        except_img_files += list(df.file.values)
    return {f: True for f in except_img_files}


def get_coco_template():
    template = {
        "info": {'year': 2021, 'version': '1.0',
                 'description': 'VIA project exported to COCO format using VGG Image Annotator (http://www.robots.ox.ac.uk/~vgg/software/via/)',
                 'contributor': '', 'url': 'http://www.robots.ox.ac.uk/~vgg/software/via/',
                 'date_created': 'Wed May 05 2021 14:18:44 GMT+0900 (한국 표준시)'}
        ,
        "licenses":
            [{'id': 0, 'name': 'Unknown License', 'url': ''}],
        "images": [
            # {
            #     "id": 1,
            #     "width": 400,
            #     "height": 400,
            #     'file_name': '1.jpg'
            # }
        ],
        "annotations": [
        ],
        "categories": [{'supercategory': 'foot', 'id': 1, 'name': 'foot'},
                       {'supercategory': 'triangle', 'id': 2, 'name': 'left_triangle'},
                       {'supercategory': 'triangle', 'id': 3, 'name': 'right_triangle'}],
    }

    return template


def make_anno(seg, width, height, image_id, id, cate_id, y=0):
    for i in range(1, len(seg[0]), 2):
        seg[0][i] += y
    rles = mask.frPyObjects(seg, height, width)
    rle = mask.merge(rles)
    bbox = list(mask.toBbox(rle))
    area = int(mask.area(rle))
    annotation = {
        "segmentation": seg,
        "area": float(area),
        "bbox": bbox,
        "iscrowd": 0,
        "image_id": image_id,
        "id": id,
        "category_id": cate_id
    }

    return annotation


def main():
    args = parse_args()
    wrong_files = {}

    if args.wrong_label_file:
        with open(args.wrong_label_file) as f:
            for line in f:
                wrong_files[line.strip()] = True

    mask_files = glob.glob(os.path.join(args.mask_dir, "*"))
    anno_id = 1
    image_id = 1
    images = []
    annos = []
    for mask_file in mask_files:
        print(image_id, mask_file)
        if os.path.splitext(os.path.basename(mask_file))[0] in wrong_files:
            print("wrong label")
            continue
        img = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        # cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        top_img = img[:img.shape[0] // 2, :]
        bot_img = img[img.shape[0] // 2:, :]

        top_tri_seg = Mask(top_img < 255 / 4).polygons().segmentation
        bot_tri_seg = Mask(bot_img < 255 / 4).polygons().segmentation
        foot_seg = Mask(img > 255 / 4 * 3).polygons().segmentation

        if len(top_tri_seg) > 1 or len(bot_tri_seg) > 1 or len(
                foot_seg) > 1 or not top_tri_seg or not bot_tri_seg or not foot_seg:
            continue

        images.append({
            "id": image_id,
            "width": img.shape[1],
            "height": img.shape[0],
            'file_name': os.path.basename(mask_file)
        })

        annos.append(make_anno(top_tri_seg, img.shape[1], img.shape[0], image_id, anno_id, 2))
        anno_id += 1
        annos.append(make_anno(bot_tri_seg, img.shape[1], img.shape[0], image_id, anno_id, 3, img.shape[0] // 2))
        anno_id += 1
        annos.append(make_anno(foot_seg, img.shape[1], img.shape[0], image_id, anno_id, 1))
        anno_id += 1
        image_id += 1

    coco = get_coco_template()
    coco["annotations"] = annos
    coco["images"] = images
    json.dump(coco, open(args.output_path, "w+"))

    print("done")


if __name__ == '__main__':
    main()
