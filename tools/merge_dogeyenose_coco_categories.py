import argparse

import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate metric of the '
                                                 'results saved in pkl format')
    parser.add_argument('--coco_files', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    coco_files = args.coco_files.split(",")

    cate_id_map = {2: 1, 3: 2}
    os.makedirs(args.output_dir, exist_ok=True)
    for coco_file in coco_files:
        coco = json.load(open(coco_file))
        for i in range(len(coco['annotations'])):
            anno_cate_id = coco['annotations'][i]["category_id"]
            if anno_cate_id in cate_id_map:
                coco['annotations'][i]["category_id"] = cate_id_map[anno_cate_id]

        coco['categories'] = [
            {'id': 1, 'name': 'dog_eye', 'supercategory': '0_true_positive', 'color': '#e23464', 'metadata': {},
             'keypoint_colors': []},
            {'id': 2, 'name': 'dog_nose', 'supercategory': '0_true_positive', 'color': '#edd273', 'metadata': {},
             'keypoint_colors': []}]
        json.dump(coco, open(os.path.join(args.output_dir, os.path.basename(coco_file)), "w+"))

    print("done")


if __name__ == '__main__':
    main()
