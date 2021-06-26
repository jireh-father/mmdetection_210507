import argparse

import json


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate metric of the '
                                                 'results saved in pkl format')
    parser.add_argument('--annotation_file', type=str, default='/media/irelin/data_disk/dataset/dog_eye_nose_detection/detect_10-13_collect/annotations.json')
    parser.add_argument('--output_path', type=str, default='/media/irelin/data_disk/dataset/dog_eye_nose_detection/detect_10-13_collect/annotations_full.json')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    anno = json.load(open(args.annotation_file))
    anno['info'] = {'year': 2021, 'version': '1.0',
                    'description': 'VIA project exported to COCO format using VGG Image Annotator (http://www.robots.ox.ac.uk/~vgg/software/via/)',
                    'contributor': '', 'url': 'http://www.robots.ox.ac.uk/~vgg/software/via/',
                    'date_created': 'Wed May 05 2021 14:18:44 GMT+0900 (한국 표준시)'}
    anno['licenses'] = [{'id': 0, 'name': 'Unknown License', 'url': ''}]

    json.dump(anno, open(args.output_path, "w+"))

    print("done")


if __name__ == '__main__':
    main()
