from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, save_result_pyplot
import glob
import os
import time
import shutil
from PIL import Image


def main():
    parser = ArgumentParser()
    parser.add_argument('--imgs', help='Image file')
    parser.add_argument('--output_dir')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    print("loading")
    model = init_detector(args.config, args.checkpoint, device=args.device)
    print("loaded")
    os.makedirs(os.path.join(args.output_dir, 'vis'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'crop'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'nodetected'), exist_ok=True)
    img_files = glob.glob(args.imgs)
    for img in img_files:
        print(os.path.basename(img))
        # test a single image
        start = time.time()
        result = inference_detector(model, img)
        print(time.time() - start)
        # show the results
        if len(result) < 1  or len(result[0]) < 1:
            shutil.copy(img, os.path.join(args.output_dir, 'nodetected'))
            continue
        output_path = os.path.join(args.output_dir, 'vis', os.path.splitext(os.path.basename(img))[0] + ".jpg")
        save_result_pyplot(model, img, result, output_path, score_thr=args.score_thr)
        im = Image.open(img).convert("RGB")
        for j, bbox in enumerate(result[0]):
            if bbox[4] < args.score_thr:
                continue
            crop_im = im.crop([int(b) for b in bbox[:-1]])
            crop_im.save(
                os.path.join(args.output_dir, 'crop', os.path.splitext(os.path.basename(img))[0] + "_{}.jpg".format(j)))


if __name__ == '__main__':
    main()
