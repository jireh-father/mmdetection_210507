import asyncio
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
import time

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    inference_detector(model, args.img)

    total = 0.
    for i in range(10):
        start = time.time()
        inference_detector(model, args.img)
        total += time.time() - start
    print(total / 10)
    # show the results
    # show_result_pyplot(model, args.img, result, score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    main(args)
