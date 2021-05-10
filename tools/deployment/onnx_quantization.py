import argparse

from onnxruntime.quantization import quantize_qat, QuantType


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate metric of the '
                                                 'results saved in pkl format')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--quant_path', type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    quantize_qat(args.model_path, args.quant_path)

    print("done")


if __name__ == '__main__':
    main()
