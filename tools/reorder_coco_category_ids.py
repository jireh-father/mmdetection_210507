import argparse

import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate metric of the '
                                                 'results saved in pkl format')
    parser.add_argument('--coco_file', type=str, default=None)
    parser.add_argument('--output_file', type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    coco_anno = json.load(open(args.coco_file))
    categories = coco_anno['categories']
    cate_map = {}
    for i, cate in enumerate(categories):
        cate_map[cate['id']] = i + 1
        coco_anno['categories'][i]['id'] = i + 1

    for i in range(len(coco_anno['annotations'])):
        cate_id = coco_anno['annotations'][i]['category_id']
        coco_anno['annotations'][i]['category_id'] = cate_map[cate_id]
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    json.dump(coco_anno, open(args.output_file, "w+"))

    print("done")


if __name__ == '__main__':
    main()
