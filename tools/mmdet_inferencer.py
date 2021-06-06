from mmdet.apis import init_detector, inference_detector, async_inference_detector
import asyncio
from argparse import ArgumentParser
import os
import glob
import time


def save_result_pyplot(model,
                       img,
                       result,
                       score_thr=0.3,
                       title='result',
                       out_file=None,
                       mask_color=None,
                       bbox_color='red',
                       text_color='red',
                       thickness=4):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        title (str): Title of the pyplot figure.
        wait_time (float): Value of waitKey param.
                Default: 0.
    """
    if hasattr(model, 'module'):
        model = model.module
    model.show_result(
        img,
        result,
        score_thr=score_thr,
        show=False,
        out_file=out_file,
        win_name=title,
        bbox_color=bbox_color,
        text_color=text_color,
        thickness=thickness,
        mask_color=mask_color)


class MmdetInferencer:
    def __init__(self, config, checkpoint, device='CPU'):
        self.model = init_detector(config, checkpoint, device=device)

    def inference(self, image_file_or_array):
        result = inference_detector(self.model, image_file_or_array)
        # result[0]: bounding box
        # result[0][0]: class 0's bboxes
        # result[0][0][0]: [x1,y1,x2,y2,score]
        # result[1]: masks
        # result[1][0]: class 0's mask
        # result[1][0][0]: masks
        return result
    #
    # def async_inference(self, image_files_or_arrays):
    #     tasks = asyncio.create_task(async_inference_detector(self.model, image_files_or_arrays))
    #     result = await asyncio.gather(tasks)
    #     print(result)


def main(args):
    print("init model")
    od = MmdetInferencer(args.config, args.checkpoint, args.device)
    if os.path.isfile(args.img_or_pattern):
        imgs = [args.img_or_pattern]
    else:
        imgs = glob.glob(args.img_or_pattern)
    # if args.async_test:
    #     od.async_inference(imgs)
    # else:

    print("inference")
    result = od.inference(imgs)
    if args.show_dir:
        print("visualize")
        os.makedirs(args.show_dir, exist_ok=True)
        for i, img in enumerate(imgs):
            out_file = os.path.join(args.show_dir, os.path.splitext(os.path.basename(img))[0] + ".jpg")
            print("{}/{} {} file is visualized to {}".format(i, len(imgs), img, out_file))
            save_result_pyplot(od.model,
                               img,
                               result[i],
                               score_thr=args.score_thr,
                               out_file=out_file,
                               # mask_color=None,
                               # bbox_color='red',
                               # text_color='red',
                               # thickness=4
                               )
    if args.benchmark:
        print("started to benchmark")
        total_exec_time = 0.
        for i in range(args.try_cnt):
            start_time = time.time()
            od.inference(imgs[0])
            total_exec_time += time.time() - start_time
        print("{} tries, avg exec time: {}".format(args.try_cnt, total_exec_time / args.try_cnt))

    print("done")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img_or_pattern', help='Image file')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument('--show_dir', default=None, type=str)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--try_cnt', default=10, type=int)
    # parser.add_argument(
    #     '--async-test',
    #     action='store_true',
    #     help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_args())
