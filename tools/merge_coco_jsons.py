import argparse

import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate metric of the '
                                                 'results saved in pkl format')
    parser.add_argument('--coco_files', type=str, default=None)
    parser.add_argument('--output_file', type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    coco_files = args.coco_files.split(",")

    image_annos = []
    for coco_file in coco_files:
        coco = json.load(open(coco_file))
        image_dict = {}
        for image in coco['images']:
            image_dict[image['id']] = {'image': image}

        for anno in coco['annotations']:
            if anno['image_id'] not in image_dict:
                print("not exist image id", anno['image_id'])
                continue
            if 'annos' not in image_dict[image['id']]:
                image_dict[image['id']]['annos'] = []
            image_dict[image['id']]['annos'].append(anno)

        image_annos += list(image_dict.values())

    new_anno_id = 1
    new_images = []
    new_annotations = []
    for i, image_anno in enumerate(image_annos):
        new_image_id = i + 1
        image_anno['image']['id'] = new_image_id
        new_images.append(image_anno['image'])

        if 'annos' not in image_anno:
            print("no anno of image", image_anno['image']['id'])
            continue
        for anno in image_anno['annos']:
            anno['id'] = new_anno_id
            anno['image_id'] = new_image_id
            new_annotations.append(anno)
            new_anno_id += 1

    new_coco = json.load(open(coco_file))
    new_coco['annotations'] = new_annotations
    new_coco['images'] = new_images
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    json.dump(new_coco, open(args.output_file, "w+"))

    print("done")


if __name__ == '__main__':
    main()
