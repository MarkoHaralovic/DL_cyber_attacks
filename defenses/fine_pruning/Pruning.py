'''
This is the implement of pruning proposed in [1].
[1] Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks. RAID, 2018.

This code is available on the following github: https://github.com/THUYimingLi/BackdoorBox
'''
import os
import torch

import numpy as np
import argparse
from utils import EfficientNetModule

EfficientNet = None
MBConvBlock = None
Conv2dStaticSamePadding = None


def count_modules(model, type_module):
    cnt = 0
    for m in model.modules():
        if isinstance(m, type_module):
            cnt += 1
    return cnt

def gen_random_base_model(model_name, cfg=None, random_init=False, override_params=None, module=None):
    model = EfficientNet.from_name(model_name, cfg=cfg, override_params=override_params)

    if random_init == True:
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data = torch.tensor(np.random.randn(m.weight.data.shape[0]), dtype=torch.float)
    return model

def is_expand_first(mbconv):
    is_expanded = False
    if mbconv._block_args.expand_ratio != 1:
        is_expanded = True
    return is_expanded

def is_skip_connected(mbconv):
    input_filters, output_filters = mbconv._block_args.input_filters, mbconv._block_args.output_filters
    is_skip_connected = False
    if mbconv.id_skip and mbconv._block_args.stride == 1 and input_filters == output_filters:
        is_skip_connected = True
    return is_skip_connected
    
def get_modules(model, type_module):
    ret = list()
    for m in model.modules():
        if isinstance(m, type_module):
            ret.append(m)
    return ret

def gen_mask_by_cfg(li_batch2d, cfg):
    li_channel_idx = list()
    for topk, m in zip(cfg, li_batch2d):
        weight_copy = m.weight.data.abs().clone()
        idx = np.squeeze(torch.argsort(weight_copy, descending=True).numpy())[:topk].tolist()
        li_channel_idx.append(idx)
    ret = li_channel_idx
    return ret

def update_bn(m1, m0, mask):
    m1.weight.data = m0.weight.data[mask].clone()
    m1.bias.data = m0.bias.data[mask].clone()
    m1.running_mean = m0.running_mean[mask].clone()
    m1.running_var = m0.running_var[mask].clone()

def gen_cfg(model, prune_ratio):
    total = 0
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
#            m.weight.data = torch.tensor(np.random.randn(m.weight.data.shape[0]), dtype=torch.float).clone()
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size
    y, i = torch.sort(bn)
    thre_index = int(total * prune_ratio)
    thre = y[thre_index]

    cfg = []
    for k, m in enumerate(model.modules()):
        if isinstance(m, torch.nn.BatchNorm2d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre).float() #TODO: CUDA
            #pruned = pruned + mask.shape[0] - torch.sum(mask)
            #m.weight.data.mul_(mask)
            #m.bias.data.mul_(mask)
            num = int(torch.sum(mask))
            if num != 0:
                cfg.append(num)
            elif num == 0:
                cfg.append(1)
            #print("layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}".
            #      format(k, mask.shape[0], int(cfg[-1])))
    return cfg

def gen_valid_cfg(model, cfg):
    cur_bn_idx = 0
    ret_cfg = cfg.copy()
    cfg = None
    li_mbconv = get_modules(model, MBConvBlock)
    cur_bn_idx += 1
    for idx, mbconv in enumerate(li_mbconv):
        start_bn_idx = cur_bn_idx
        if is_expand_first(mbconv) == True:
            cur_bn_idx += 1 #expand block
        #depthwise
        ret_cfg[cur_bn_idx] = ret_cfg[cur_bn_idx - 1]
        cur_bn_idx += 1
        
        if is_skip_connected(mbconv):
            ret_cfg[cur_bn_idx] = ret_cfg[start_bn_idx - 1]
        cur_bn_idx += 1
    return ret_cfg

def get_conv_module(m):
    ret = m
    if hasattr(m, "conv"):
        ret = m.conv
    assert isinstance(ret, torch.nn.Conv2d) is True
    return ret

def update_parameters(new_model, old_model, cfg):
    li_new_conv2d = get_modules(new_model, Conv2dStaticSamePadding)
    li_new_bn = get_modules(new_model, torch.nn.BatchNorm2d)
    li_new_mbconv = get_modules(new_model, MBConvBlock)
    
    li_old_conv2d = get_modules(old_model, Conv2dStaticSamePadding)
    li_old_bn = get_modules(old_model, torch.nn.BatchNorm2d)
    li_old_mbconv = get_modules(old_model, MBConvBlock)
    
    li_channel_idx = gen_mask_by_cfg(li_old_bn, cfg)
    #print(li_channel_idx)
    ####### Not MBConvBlock ###########
    
    first_new_conv2d = get_conv_module(li_new_conv2d[0])
    last_new_conv2d = get_conv_module(li_new_conv2d[-1])
    
    first_old_conv2d = get_conv_module(li_old_conv2d[0])
    last_old_conv2d = get_conv_module(li_old_conv2d[-1]) 
    
    first_new_bn, last_new_bn = li_new_bn[0], li_new_bn[-1]
    first_old_bn, last_old_bn = li_old_bn[0], li_old_bn[-1]
    
    ####### first layer ###########
    
    w1 = first_old_conv2d.weight.data[li_channel_idx[0], :, :, :].clone()
    first_new_conv2d.weight.data = w1.clone()
    update_bn(first_new_bn, first_old_bn, li_channel_idx[0])
    
    cur_bn_idx = 1
    cur_conv_idx = 1
    for idx, (new_mbconv, old_mbconv) in enumerate(zip(li_new_mbconv, li_old_mbconv)):
        #print(idx)
        start_bn_idx = cur_bn_idx
        if is_expand_first(new_mbconv) == True:
            new_expand_conv = get_conv_module(new_mbconv._expand_conv)
            old_expand_conv = get_conv_module(old_mbconv._expand_conv)
            w1 = old_expand_conv.weight.data[:, li_channel_idx[cur_bn_idx - 1], :, :].clone()
            w1 = w1[li_channel_idx[cur_bn_idx], :, :, :].clone()
            new_expand_conv.weight.data = w1.clone()
            
            new_bn0 = new_mbconv._bn0
            old_bn0 = old_mbconv._bn0
            update_bn(new_bn0, old_bn0, li_channel_idx[cur_bn_idx])
            cur_bn_idx += 1
        ##############################################################################
        
        new_depthwise_conv = get_conv_module(new_mbconv._depthwise_conv)
        old_depthwise_conv = get_conv_module(old_mbconv._depthwise_conv)
        
        
        #depthwise [channels, 1, x_filter, y_filter]
        li_channel_idx[cur_bn_idx] = li_channel_idx[cur_bn_idx - 1] # align same order in depthwise
        w1 = old_depthwise_conv.weight.data[li_channel_idx[cur_bn_idx], :, :, :].clone()
        new_depthwise_conv.weight.data = w1.clone()
        
        new_bn1 = new_mbconv._bn1
        old_bn1 = old_mbconv._bn1
        update_bn(new_bn1, old_bn1, li_channel_idx[cur_bn_idx])
        
        #...........................................................................
        #reduce
        new_se_reduce = get_conv_module(new_mbconv._se_reduce)
        old_se_reduce = get_conv_module(old_mbconv._se_reduce)
        w1 = old_se_reduce.weight.data[:, li_channel_idx[cur_bn_idx], :, :].clone()
        new_se_reduce.weight.data = w1.clone()
        new_se_reduce.bias.data = old_se_reduce.bias.data.clone()
                
        #expansion
        new_se_expand = get_conv_module(new_mbconv._se_expand)
        old_se_expand = get_conv_module(old_mbconv._se_expand)
        
        
        w1 = old_se_expand.weight.data[li_channel_idx[cur_bn_idx], :, :, :].clone()
        #w1 = old_se_expand.weight.data.clone()
        new_se_expand.weight.data = w1.clone() #elementwise multiplication by after bn1
        new_se_expand.bias.data = old_se_expand.bias.data[li_channel_idx[cur_bn_idx]].clone()
        #...........................................................................
        cur_bn_idx += 1
        ##############################################################################
        
        
        new_project_conv = get_conv_module(new_mbconv._project_conv)
        old_project_conv = get_conv_module(old_mbconv._project_conv)
        
        #last
        in_channel = li_channel_idx[cur_bn_idx - 1]
        if is_skip_connected(new_mbconv):
            li_channel_idx[cur_bn_idx] = li_channel_idx[start_bn_idx - 1]
            
        out_channel = li_channel_idx[cur_bn_idx]
        
        w1 = old_project_conv.weight.data[out_channel, :, :, :].clone()
        w1 = w1.data[:, in_channel, :, :].clone()
        new_project_conv.weight.data = w1.clone()
        
        
        new_bn2 = new_mbconv._bn2
        old_bn2 = old_mbconv._bn2
        
        update_bn(new_bn2, old_bn2, out_channel)
        cur_bn_idx += 1
        ##############################################################################
    
    assert len(cfg) - 1 == cur_bn_idx
    #last conv
    
    in_channel = li_channel_idx[cur_bn_idx - 1]
    out_channel = li_channel_idx[cur_bn_idx]
    w1 = last_old_conv2d.weight.data[:, in_channel, :, :].clone()
    w1 = w1[out_channel, :, :, :].clone()
    last_new_conv2d.weight.data = w1.clone()
    
    
    update_bn(last_new_bn, last_old_bn, out_channel)
    #last linear layer
    
    li_new_linear = get_modules(new_model, torch.nn.Linear)
    li_old_linear = get_modules(old_model, torch.nn.Linear)
    assert len(li_new_linear) == 1
    last_new_linear, last_old_linear = li_new_linear[-1], li_old_linear[-1]
    #print(old_last_linear.weight.shape, new_last_linear.weight.shape)
    in_channel = out_channel
    w1 = last_old_linear.weight.data[:, in_channel].clone()
    last_new_linear.weight.data = w1.clone()
    b1 = last_old_linear.bias.data.clone()
    last_new_linear.bias.data = b1.clone()

def get_parameters(model):
    ret = sum([p.view(-1).shape[0] for p in model.parameters()])
    return ret

def prune_model(name, path, prune_ratio, override_params, module):
    device = torch.device('cpu')
    loaded = EfficientNet.from_name_pruned(name, state_dict_path=path, override_params=override_params)

    state_dict = torch.load(path, map_location=device)
    loaded.load_state_dict(state_dict)
    old_model = get_modules(loaded, EfficientNet)[0].cpu()

    cfg = gen_cfg(old_model, prune_ratio=prune_ratio)
    valid_cfg = gen_valid_cfg(old_model, cfg)
    new_model = gen_random_base_model(name, cfg=valid_cfg, random_init=False, override_params=override_params, module=module)
    update_parameters(new_model, old_model, cfg=valid_cfg)

    old_num_p = get_parameters(old_model)
    new_num_p = get_parameters(new_model)
    print("parameters: {} -> {}. ({}%)".format(old_num_p, new_num_p, new_num_p / old_num_p))
    return new_model 

def set_repo(repo):
    global EfficientNet
    global MBConvBlock
    global Conv2dStaticSamePadding
    
    module = EfficientNetModule(repo)
    EfficientNet = module.EfficientNet(need_hook=True)
    MBConvBlock = module.MBConvBlock()
    Conv2dStaticSamePadding = module.Conv2dStaticSamePadding()
    return module

def main(args):
    efficientnet_module = set_repo(args.efficientnet_repo)
    new_model = prune_model(args.efficientnet_kind, args.load_path, args.prune_ratio, dict(num_classes=args.num_classes), efficientnet_module)
    state_dict = new_model.state_dict()
    torch.save(state_dict, args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="efficientnet pruner")
    parser.add_argument("--efficientnet-kind", type=str, default="efficientnet-b0",
                        help="efficientnet kind (ex: efficientnet-b0, effcientnet-b1)")

    parser.add_argument("--efficientnet-repo", type=str, default="yet",
                        help="efficientnet kind (ex: yet(Yet Another EfficientNet-Pytorch), ept(Efficientnet-PyTorch))")

    parser.add_argument("--num-classes", type=int, default=1000,
                        help="number of classes")

    parser.add_argument("--save-path", type=str, default=None,
                        help="save path")

    parser.add_argument("--load-path", type=str, default=None,
                        help="load path")
    
    parser.add_argument("--prune-ratio", type=float, default=0.5,
                        help="Magnitude pruning ratio")
    args = parser.parse_args()
    main(args)