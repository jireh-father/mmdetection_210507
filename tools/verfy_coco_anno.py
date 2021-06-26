import argparse
import json
import os

from pycocotools import mask


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate metric of the '
                                                 'results saved in pkl format')
    parser.add_argument('--annotation_file', type=str,
                        default='/media/irelin/data_disk/dataset/dog_eye_nose_detection/detect_10-13_collect/eye_annotations.json')
    parser.add_argument('--output_path', type=str,
                        default='/media/irelin/data_disk/dataset/dog_eye_nose_detection/detect_10-13_collect/eye_annotations_verified.json')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
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

    for idx, image_id in enumerate(images_dict):
        # if idx < 2090:
        #     continue
        print(idx, len(images_dict))
        image_info = images_dict[image_id]
        if image_id not in image_anno_dict:
            print(image_id, "skip")
            continue
        tmp_annos = image_anno_dict[image_id]
        w = image_info['width']
        h = image_info['height']

        for tmp_anno in tmp_annos:
            if len(tmp_anno['segmentation']) > 1:
                seg = tmp_anno['segmentation']
            else:
                seg = tmp_anno['segmentation'][0]

            for x_i in range(0, len(seg), 2):
                y_i = x_i + 1
                seg[x_i] = min(max(0, seg[x_i]), w - 1)
                seg[y_i] = min(max(0, seg[y_i]), h - 1)

            seg = [seg]

            bbox = tmp_anno['bbox']
            x = bbox[0]
            y = bbox[1]
            bw = bbox[2]
            bh = bbox[3]

            if x < 0:
                bw += x
                x = 0
            if y < 0:
                bh += y
                y = 0

            if x + bw > w:
                bw -= ((x + bw) - w)

            if y + bh > h:
                bh -= ((y + bh) - h)

            bbox[0] = x
            bbox[1] = y
            bbox[2] = bw
            bbox[3] = bh

            rles = mask.frPyObjects(seg, h, w)
            rle = mask.merge(rles)
            area = int(mask.area(rle))

            new_anno = tmp_anno
            new_anno['segmentation'] = seg
            new_anno['bbox'] = bbox
            new_anno['area'] = area
            new_annos.append(new_anno)
            new_images.append(image_info)

    anno["images"] = new_images
    anno["annotations"] = new_annos
    json.dump(anno, open(args.output_path, "w+"))

    print("done")


if __name__ == '__main__':
    main()
