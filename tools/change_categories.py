import argparse

import json
import os
import glob


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate metric of the '
                                                 'results saved in pkl format')
    parser.add_argument('--coco_files_or_pattern', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--categories_json', type=str,
                        default='[{"id": 1, "name": "eyes", "supercategory": "0_true_position"}]')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    coco_files_or_patterns = args.coco_files_or_pattern.split(",")
    if os.path.isfile(coco_files_or_patterns[0]):
        coco_files = coco_files_or_patterns
    else:
        coco_files = glob.glob(args.coco_files_or_pattern)

    os.makedirs(args.output_dir, exist_ok=True)
    for coco_file in coco_files:
        coco = json.load(open(coco_file))
        coco['categories'] = json.loads(args.cateogries_json)
        output_path = os.path.join(args.output_dir, os.path.basename(coco_file))
        json.dump(coco, open(output_path, "w+"))

    print("done")


if __name__ == '__main__':
    main()
