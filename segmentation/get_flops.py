import argparse

from mmcv import Config
from mmcv.cnn import get_model_complexity_info
from mmseg.models import build_segmentor
import denseclip

from fvcore.nn import FlopCountAnalysis

import torch
from numbers import Number
from typing import Any, Callable, List, Optional, Union
from numpy import prod
import numpy as np
from fvcore.nn import FlopCountAnalysis
from mmseg.datasets import build_dataset


def calc_flops(model, img_size=224):
    with torch.no_grad():
        x = torch.randn(1, 3, img_size, img_size).cuda()
        fca1 = FlopCountAnalysis(model, x)
        print('backbone:', fca1.total(module_name="backbone")/1e9)
        try:
            print('text_encoder:', fca1.total(module_name="text_encoder")/1e9)
            print('context_decoder:', fca1.total(module_name="context_decoder")/1e9)
        except:
            pass
        
        try:
            print('neck:', fca1.total(module_name="neck")/1e9)
        except:
            pass
        print('decode_head:', fca1.total(module_name="decode_head")/1e9)
        flops1 = fca1.total()
        print("#### GFLOPs: {:.1f}".format(flops1 / 1e9))
    return flops1 / 1e9

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--fvcore',
        action='store_true', default=False)
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1024, 1024],
        help='input image size')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    
    

    if 'DenseCLIP' in cfg.model.type:
        datasets = [build_dataset(cfg.data.train)]
        cfg.model.class_names = list(datasets[0].CLASSES)
    
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')).cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))
        
    if args.fvcore:
        flops = calc_flops(model, input_shape[1])
        
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print('number of params:', f'{n_parameters:.1f}')
        if hasattr(model, 'text_encoder'):
            n_parameters_text = sum(p.numel() for p in model.text_encoder.parameters() if p.requires_grad) / 1e6
            print('param without text encoder:', n_parameters-n_parameters_text)
        if hasattr(model, 'context_decoder'):
            n_parameters_text = sum(p.numel() for p in model.context_decoder.parameters() if p.requires_grad) / 1e6
            print('param context:', n_parameters_text)
    else:
        flops, params = get_model_complexity_info(model, input_shape)
        split_line = '=' * 30
        print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
            split_line, input_shape, flops, params))
        print('!!!Please be cautious if you use the results in papers. '
              'You may need to check if all ops are supported and verify that the '
              'flops computation is correct.')
        

if __name__ == '__main__':
    main()
