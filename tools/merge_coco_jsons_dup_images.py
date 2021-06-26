import argparse
import glob
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate metric of the '
                                                 'results saved in pkl format')
    parser.add_argument('--coco_files_or_pattern', type=str,
                        default='/media/irelin/data_disk/dataset/dog_eye_nose_detection/detect_10-13_collect/cate_fixed_annos/eye_annotations_verified.json,/media/irelin/data_disk/dataset/dog_eye_nose_detection/detect_10-13_collect/cate_fixed_annos/nose_annotations_verified.json')
    parser.add_argument('--output_file', type=str,
                        default='/media/irelin/data_disk/dataset/dog_eye_nose_detection/detect_10-13_collect/dog_eye_nose_annotations.json')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    coco_files_or_patterns = args.coco_files_or_pattern.split(",")
    if os.path.isfile(coco_files_or_patterns[0]):
        coco_files = coco_files_or_patterns
    else:
        coco_files = glob.glob(args.coco_files_or_pattern)

    image_dict = {}
    for coco_file in coco_files:
        coco = json.load(open(coco_file))
        image_id_dict = {}
        for image in coco['images']:
            fname = image['file_name']
            if fname not in image_dict:
                image_dict[fname] = {'image': image}
            image_id_dict[image['id']] = fname

        for anno in coco['annotations']:
            tmp_im_id = anno['image_id']
            if tmp_im_id not in image_id_dict:
                print("not exist image id", anno['image_id'])
                continue

            if 'annos' not in image_dict[image_id_dict[tmp_im_id]]:
                image_dict[image_id_dict[tmp_im_id]]['annos'] = []
            image_dict[image_id_dict[tmp_im_id]]['annos'].append(anno)

    image_annos = list(image_dict.values())

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
