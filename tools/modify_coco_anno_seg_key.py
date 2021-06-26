import argparse

import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate metric of the '
                                                 'results saved in pkl format')
    parser.add_argument('--annotation_file', type=str,
                        default='/media/irelin/data_disk/dataset/dog_eye_nose_detection/detect_10-13_collect/annotations.json')
    parser.add_argument('--output_path', type=str,
                        default='/media/irelin/data_disk/dataset/dog_eye_nose_detection/detect_10-13_collect/annotations_fixed_seg_key.json')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    anno = json.load(open(args.annotation_file))

    images = anno["images"]
    print("total", len(images))

    for i in range(len(anno["annotations"])):
        if 'segmentations' not in anno["annotations"][i]:
            continue
        anno["annotations"][i]['segmentation'] = anno["annotations"][i]['segmentations']
        del anno["annotations"][i]['segmentations']
    json.dump(anno, open(args.output_path, "w+"))

    print("done")


if __name__ == '__main__':
    main()
