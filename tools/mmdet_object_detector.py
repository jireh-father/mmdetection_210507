from mmdet.apis import init_detector, inference_detector, async_inference_detector
import asyncio
from argparse import ArgumentParser
import os
import glob


class MmdetObjectDetector:
    def __init__(self, config, checkpoint, device='CPU'):
        self.model = init_detector(config, checkpoint, device=device)

    def inference(self, image_file_or_array):
        result = inference_detector(self.model, image_file_or_array)
        print(len(result))
        print(len(result[0]))
        print(len(result[1]))
    #
    # def async_inference(self, image_files_or_arrays):
    #     tasks = asyncio.create_task(async_inference_detector(self.model, image_files_or_arrays))
    #     result = await asyncio.gather(tasks)
    #     print(result)

def main(args):
    od = MmdetObjectDetector(args.config, args.checkpoint, args.device)
    if os.path.isfile(args.img_or_pattern):
        imgs = args.img_or_pattern
    else:
        imgs = glob.glob(args.img_or_pattern)
    # if args.async_test:
    #     od.async_inference(imgs)
    # else:
    od.inference(imgs)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img_or_pattern', help='Image file')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    # parser.add_argument(
    #     '--async-test',
    #     action='store_true',
    #     help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_args())
