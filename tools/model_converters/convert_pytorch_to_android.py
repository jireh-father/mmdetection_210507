import torch
import torch.nn as nn
import argparse
import os
from mmdet.apis import inference_detector, init_detector


model_classifier_map = {
    'alexnet': ['classifier', 6],
    'vgg': ['classifier', 6],
    'mobilenet': ['classifier', 1],
    'mnasnet': ['classifier', 6],
    'resnet': ['fc'],
    'inception': ['fc'],
    'googlenet': ['fc'],
    'shufflenet': ['fc'],
    'densenet': ['classifier'],
    'resnext': ['fc'],
    'wide_resnet': ['fc'],
    'efficientnet': ['_fc'],
    'bagnet': ['fc'],
    'rexnet': ['output', 1],
}


def init_model(model_name, num_classes):
    if model_name.startswith("efficientnet"):
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
        return model

    from torchvision import models
    for m_key in model_classifier_map:
        if m_key in model_name:
            model_fn = getattr(models, model_name)
            cls_layers = model_classifier_map[m_key]

            if model_name.startswith("inception"):
                # input_size = 299
                model = model_fn(aux_logits=False)
            else:
                # input_size = 224
                model = model_fn()

            if len(cls_layers) == 1:
                in_features = getattr(model, cls_layers[0]).in_features
                setattr(model, cls_layers[0], nn.Linear(in_features, num_classes))
            else:
                classifier = getattr(model, cls_layers[0])
                in_features = classifier[cls_layers[1]].in_features
                classifier[cls_layers[1]] = nn.Linear(in_features, num_classes)
            return model


def main(args):
    device = 'cpu'
    model = init_model(args.model_name, args.num_classes)
    if args.model_name.startswith("efficientnet"):
        model.set_swish(memory_efficient=False)

    checkpoint_dict = torch.load(args.model_path, map_location=device)
    pretrained_dict = checkpoint_dict['state_dict']

    try:
        model.load_state_dict(pretrained_dict)
    except:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(pretrained_dict)
        model = model.module

    model = init_detector(args.config, args.checkpoint, device=args.device)

    model = model.to(device)
    model.eval()

    example = torch.rand(1, 3, args.input_size, args.input_size)
    ret = model(example)
    print(ret, ret.shape)
    traced_script_module = torch.jit.trace(model, example)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    traced_script_module.save(args.output_path)
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--output_path', default=None, type=str)
    parser.add_argument('--model_name', default=None, type=str)
    parser.add_argument('--num_classes', default=None, type=int)
    parser.add_argument('--input_size', default=224, type=int)

    main(parser.parse_args())
