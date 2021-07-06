import asyncio
import glob
import time
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--imgs', help='Image file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    img_files = glob.glob(args.imgs)

    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    inference_detector(model, img_files[0])
    total = 0.
    for im in img_files:
        start = time.time()
        inference_detector(model, im)
        total += time.time() - start
    # show the results
    print(total / len(img_files))


async def async_main(args):
    # build the model from a config file and a checkpoint file
    img_files = glob.glob(args.imgs)
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    inference_detector(model, img_files[0])
    tasks = asyncio.create_task(async_inference_detector(model, img_files))
    start = time.time()
    await asyncio.gather(tasks)
    total = time.time() - start
    # show the results
    print(total / len(img_files))


if __name__ == '__main__':
    args = parse_args()
    start = time.time()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
    print(time.time() - start)
