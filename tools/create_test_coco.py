import argparse

import json
import os
import glob
import pandas as pd
import sys
from pycocotools import mask


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate metric of the '
                                                 'results saved in pkl format')
    parser.add_argument('--image_dir', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
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


def main():
    args = parse_args()

    im_id = 1
    images = []
    for i, image_file in enumerate(glob.glob(os.path.join(args.image_dir, "*"))):
        if i % 10 == 0:
            print(i, len(image_file))

        im_data = {
            "id": im_id,
            "width": 1280,
            "height": 720,
            'file_name': os.path.basename(image_file)
        }
        images.append(im_data)
        im_id += 1

    coco = get_coco_template()
    coco["images"] = images
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    json.dump(coco, open(args.output_path, "w+"))

    print("done")


if __name__ == '__main__':
    main()
