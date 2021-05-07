from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, save_result_pyplot
import glob
import os
import time
import shutil
from PIL import Image
import traceback


def main():
    parser = ArgumentParser()
    parser.add_argument('--img_dirs', help='Image file')
    parser.add_argument('--output_dir')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--margin_ratio', type=float, default=None, help='bbox score threshold')
    parser.add_argument(
        '--crop_square', action='store_true', default=False, help='bbox score threshold')
    parser.add_argument(
        '--use_only_crop', action='store_true', default=False, help='bbox score threshold')

    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    for i, img_dir in enumerate(glob.glob(os.path.join(args.img_dirs, "*"))):
        if args.use_only_crop:
            os.makedirs(os.path.join(args.output_dir, os.path.basename(img_dir)), exist_ok=True)
        else:
            os.makedirs(os.path.join(args.output_dir, os.path.basename(img_dir), 'vis'), exist_ok=True)
            os.makedirs(os.path.join(args.output_dir, os.path.basename(img_dir), 'crop'), exist_ok=True)
            os.makedirs(os.path.join(args.output_dir, os.path.basename(img_dir), 'nodetected'), exist_ok=True)
        img_files = glob.glob(os.path.join(img_dir, "*"))
        # if i < 366:
        #     print("skip")
        #     continue
        for j, img in enumerate(img_files):
            #     if i == 366 and j < 140:
            #         print("skip")
            #         continue
            print(i, j, len(img_files), os.path.basename(img), os.path.basename(os.path.dirname(img)))
            if args.use_only_crop:
                crop_output_path = os.path.join(args.output_dir, os.path.basename(img_dir),
                                                os.path.splitext(os.path.basename(img))[0] + "_*.jpg")
                if len(glob.glob(crop_output_path)) > 0:
                    print("skip")
                    continue
            else:
                output_path = os.path.join(args.output_dir, os.path.basename(img_dir), 'vis',
                                           os.path.splitext(os.path.basename(img))[0] + ".jpg")
                crop_output_path = os.path.join(args.output_dir, os.path.basename(img_dir), 'crop',
                                                os.path.splitext(os.path.basename(img))[0] + "_*.jpg")
                if os.path.isfile(output_path) and len(glob.glob(crop_output_path)) > 0:
                    print("skip")
                    continue
            # test a single image
            start = time.time()
            try:
                result = inference_detector(model, img)
            except:
                traceback.print_exc()
                continue
            print(time.time() - start)
            # show the results
            if len(result) < 1 or len(result[0]) < 1:
                if args.use_only_crop:
                    if not os.path.isdir(os.path.join(args.output_dir, os.path.basename(img_dir) + "_nodetected")):
                        os.makedirs(os.path.join(args.output_dir, os.path.basename(img_dir) + "_nodetected"),
                                    exist_ok=True)
                    shutil.copy(img, os.path.join(args.output_dir, os.path.basename(img_dir) + "_nodetected"))
                else:
                    shutil.copy(img, os.path.join(args.output_dir, os.path.basename(img_dir), 'nodetected'))
                continue

            if not args.use_only_crop:
                save_result_pyplot(model, img, result, output_path, score_thr=args.score_thr)
            try:
                im = Image.open(img).convert("RGB")
            except:
                traceback.print_exc()
                continue
            for j, bbox in enumerate(result[0]):
                if bbox[4] < args.score_thr:
                    continue
                im_width, im_height = im.size
                if args.crop_square:
                    if args.margin_ratio and args.margin_ratio > 0:
                        x1, y1, x2, y2, _ = bbox
                        w = x2 - x1
                        h = y2 - y1
                        w_margin = w * args.margin_ratio
                        h_margin = h * args.margin_ratio
                        x1 -= w_margin
                        x2 += w_margin
                        y1 -= h_margin
                        y2 += h_margin
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(x2, im_width)
                        y2 = min(y2, im_height)
                    else:
                        x1, y1, x2, y2 = [int(b) for b in bbox[:-1]]
                    w = x2 - x1
                    h = y2 - y1
                    if w > h:
                        diff = w - h
                        half, remain = divmod(diff, 2)
                        y1 -= (half + remain)
                        y2 += half
                        y1 = max(0, y1)
                        y2 = min(y2, im_height)
                    elif w < h:
                        diff = h - w
                        half, remain = divmod(diff, 2)
                        x1 -= (half + remain)
                        x2 += half
                        x1 = max(0, x1)
                        x2 = min(x2, im_width)
                    crop_im = im.crop((x1, y1, x2, y2))
                else:
                    if args.margin_ratio and args.margin_ratio > 0:
                        x1, y1, x2, y2, _ = bbox
                        w = x2 - x1
                        h = y2 - y1
                        w_margin = w * args.margin_ratio
                        h_margin = h * args.margin_ratio
                        x1 -= w_margin
                        x2 += w_margin
                        y1 -= h_margin
                        y2 += h_margin
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(x2, im_width)
                        y2 = min(y2, im_height)

                        crop_im = im.crop((x1, y1, x2, y2))
                    else:
                        crop_im = im.crop([int(b) for b in bbox[:-1]])
                if args.use_only_crop:
                    crop_im.save(
                        os.path.join(args.output_dir, os.path.basename(img_dir),
                                     os.path.splitext(os.path.basename(img))[0] + "_{}.jpg".format(j)))

                else:
                    crop_im.save(
                        os.path.join(args.output_dir, os.path.basename(img_dir), 'crop',
                                     os.path.splitext(os.path.basename(img))[0] + "_{}.jpg".format(j)))


if __name__ == '__main__':
    main()
