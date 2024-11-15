# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import sys
import os
import time

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from utils.config import img_param_init, set_random_seed
from utils.prepare_data_dg_clip import *
from collections import OrderedDict
import argparse
from nets.models import ClipModelat
from nets.plan import *
from torch.utils.tensorboard import SummaryWriter
from utils.lr_scheduler import *
import torch.optim as optim
from tqdm import tqdm
from math import ceil
import numpy as np
import pickle
import torch

from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count_table, flop_count_str
from data.Fourier_Aug import Shuffle_Batch_Data, Batch_FFT2_Amp_MixUp
from typing import List, Any
from numbers import Number


def is_last_batch(total_data,  current_index):
    current_index += 1
    return current_index == total_data

def train(a_iter, wi, args, model, data_loader, optimizer, scheduler, device,communication_i):
    model.train()
    batch_id = 0



    for batch in tqdm(data_loader):


        model.batch_id = batch_id
        image,  label = batch
        image = image.to(device)
        # a = count_flops(model,image)
        label = label.to(device)

        # if a_iter % 2 != 0:
        #     shuffle_imgs = Shuffle_Batch_Data(image)
        #     image = Batch_FFT2_Amp_MixUp(image, shuffle_imgs)

        loss = model(image,label, training=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if a_iter < args.WARMUP_EPOCH and a_iter>0:
        #     scheduler.step()

        if batch_id + 1 == len(data_loader):
            print(f"Current lr: {optimizer.param_groups[0]['lr']:.4e}")
        batch_id += 1


def test(args, model, data_loader, device):
    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for batch in tqdm(data_loader):
            image, label = batch
            image = image.to(device)
            label = label.to(device)
            y_true = []
            y_pred = []

            output,image_features = model(image)
            pred = output.max(1)[1]
            matches = pred.eq(label).float()
            correct += int(matches.sum().item())
            total += label.shape[0]

            acc = 100.0 * correct / total

            # err = 100.0 - acc
            # macro_f1 = 100.0 * f1_score(
            #     y_true,
            #     y_pred,
            #     average="macro",
            #     labels=np.unique(y_true)
            # )

        return acc


def communication(args, server_model, models, client_weights):
    client_num = len(models)

    with torch.no_grad():
        for key in server_model.state_dict().keys():

            temp = torch.zeros_like(server_model.state_dict()[
                                        key], dtype=torch.float16)
            if 'prompt_learner' in key or "UnifiedPrompt" in key:
                for client_idx in range(client_num):
                    # 判断prompt_learner里的是VPT还是adapter
                    # adapter 共享参数
                    if client_idx not in args.test_envs:
                        # 每个客户端将自己的prompt传给server
                        if "VPT" in key:
                            if client_idx - int(args.test_envs[0]) < 0:
                                VPT_i = client_idx
                            else:
                                VPT_i = client_idx - 1
                            # VPT_i = client_idx
                            if str(VPT_i) in key and "VPTcompound_prompts_vision" and "VPTcompound_prompts_text." not in key:
                                temp = models[client_idx].state_dict()[key]
                            else:
                                if "VPTcompound_prompts_vision." + str(VPT_i) in key:
                                    temp = models[client_idx].state_dict()[key]
                                if "VPTcompound_prompts_text." + str(VPT_i) in key:
                                    temp = models[client_idx].state_dict()[key]

                        else:

                            temp += client_weights[client_idx] * \
                                    models[client_idx].state_dict()[key]

                server_model.state_dict()[key].data.copy_(temp)

                for client_idx in range(client_num):
                    if client_idx not in args.test_envs:
                        models[client_idx].state_dict()[key].data.copy_(
                            server_model.state_dict()[key])

    return server_model, models


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='office-home')
    parser.add_argument('--lr1', type=float, default=0.005, help='learning rate')
    parser.add_argument('--lr2', type=float, default=0.005, help='learning rate')
    parser.add_argument('--datapercent', type=float,
                        default=6e-1, help='data percent to use')
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--root_dir', type=str, default='/home/gongshuai/code/FedDG/data/')
    parser.add_argument('--iters', type=int, default=300,
                        help='iterations for communication')
    parser.add_argument('--wk_iters', type=int, default=1,
                        help='optimization iters in local worker between communication')
    parser.add_argument('--mode', type=str, default='FedAtImg')
    parser.add_argument('--backbone_name', type=str, default='ViT-B/16',
                        help='[RN50 | RN101 | RN50x4 | RN50x16 | RN50x64 | ViT-B/32 | ViT-B/16 | ViT-L/14 | ViT-L/14@336px]')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--n_clients', type=int, default=4)
    parser.add_argument('--test_envs', type=int, nargs='+', default=[1])
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--optimizers', type=str, default='SGD', help='Adam or SGD')

    parser.add_argument('--WARMUP_EPOCH', type=int, default=2)
    parser.add_argument('--WARMUP_CONS_LR', type=float, default=1e-05)
    #   Default settings for MaPLe,Coop and CoCoop
    parser.add_argument('--N_CTX', type=int, default=8)  # prompt length
    parser.add_argument('--CTX_INIT', type=str, default='a photo of a')
    parser.add_argument('--INPUT_SIZE', type=tuple, default=(224, 224))
    parser.add_argument('--PROMPT_DEPTH', type=int, default=12)
    parser.add_argument('--PROMPT_DEPTH_VISION', type=int, default=12)
    parser.add_argument('--N_CTX_VISION', type=int, default=8)
    parser.add_argument('--N_CTX_TEXT', type=int, default=8)
    parser.add_argument('--PROMPT_DEPTH_TEXT', type=int, default=12)
    parser.add_argument('--cos_scale', type=int, default=1)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--if_loss_domain', type=bool, default=False)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--num_tarin_client_prompts', type=int, default=5)
    parser.add_argument('--num_tarin_adapters', type=int, default=5)
    parser.add_argument('--prototype_path', type=str, default="E:/FedDG/prototype/")
    parser.add_argument('--text_embedding_path', type=str, default="E:/FedDG/text_embedding/")
    parser.add_argument('--a_t', type=float, default=0)
    parser.add_argument('--b_t', type=float, default=0)
    parser.add_argument('--c', type=float, default=0)
    parser.add_argument('--d', type=float, default=0)
    parser.add_argument('--e', type=float, default=0)

    parser.add_argument('--lambada_', type=float, default=0)

    parser.add_argument('--num_shots', type=int, default=16)
    parser.add_argument('--save_path', type=str, default="E:/FedDG/save_models/")


    args = parser.parse_args()
    args.random_state = np.random.RandomState(1)
    set_random_seed(args.seed)

    args = img_param_init(args)
    os.makedirs('../data/', exist_ok=True)

    FedCLIP = ClipModelat(
        args.backbone_name, imgadpy=True, freezepy=True)

    train_loaders, val_loaders, test_loaders = get_data(
        args.dataset)(args, FedCLIP)

    # build server model
    class_names = test_loaders[(args.test_envs)[0]].batch_sampler.sampler.data_source.data.classes

    args.classnames = class_names
    # 1.load the modify clip model
    # 2.design details
    design_details = {"trainer": 'MaPLe',
                      "vision_depth": args.PROMPT_DEPTH_VISION,
                      "vision_ctx": args.N_CTX_VISION,
                      "language_depth": args.N_CTX,
                      "language_ctx": args.N_CTX
                      }

    # build the server model
    clip_model = load_clip_to_cpu(args)
    args.communication_i = 0
    args.client_i = 0
    logrecord = ''

    server_model = CustomCLIP(args, class_names, clip_model)
    server_model = server_model.to(device=device)

    client_num = len(test_loaders)
    sclient_num = client_num - len(args.test_envs)
    client_weights = [1 / sclient_num for i in range(client_num)]
    models = [copy.deepcopy(server_model) for idx in range(client_num)]

    name_to_update = "prompt_learner"

    for i in range(client_num):
        for name, param in models[i].named_parameters():
            if "prompt_learner" in name and "ZS_image_encoder" not in name:
                param.requires_grad_(True)

            elif "UnifiedPrompt" in name:
                param.requires_grad_(True)

            else:
                param.requires_grad_(False)
        models[i] = models[i].to(device)
        # Double check
    param_dict = OrderedDict()
    for i in range(client_num):
        param_dict[i] = []
        for name, param in models[i].named_parameters():
            if param.requires_grad:
                param_dict[i].append(name)


    best_changed = False

    best_acc = [0. for j in range(client_num)]
    finalrecord = ''
    logrecord = ''



    for a_iter in range(args.iters):
        start_time = time.time()

        optimizers = [optim.SGD(
            [{'params': models[idx].parameters()}],
            lr=args.lr2,
            momentum=0.9,
            weight_decay=0.0005,
            dampening=0,
            nesterov=False,
        ) for idx in range(client_num)]
        
        schedulerss = [0] * 4

        wk_iters = args.wk_iters

        for wi in range(wk_iters):
            num_models = 0
            print("============ Train epoch {} ============".format(
                wi + a_iter * args.wk_iters))
            logrecord += 'Train epoch:%d\n' % (wi + a_iter * args.wk_iters)
            for client_idx, model in enumerate(models):

                # prototype_path = args.prototype_path + args.dataset + "/" + str(client_idx) + ".pkl"
                # # model.prototpye = pkl_load(prototype_path)
                #
                text_embedding_path = args.text_embedding_path + args.dataset + ".pkl"
                model.fixed_embeddings = pkl_load(text_embedding_path)

                if client_idx in args.test_envs:
                    pass
                else:
                
                    num_models += 1

                    communication_i = a_iter
                    client_i = client_idx
                    model.client_i = client_i
                    model.communication_i = communication_i

                    train(
                        a_iter, wi, args, model, train_loaders[client_idx], optimizers[client_idx],
                        schedulerss[client_idx], device,communication_i)

        with torch.no_grad():


            server_model, models = communication(
                args, server_model, models, client_weights)



            val_acc_list = [0. for j in range(client_num)]

            for client_idx, model in enumerate(models):

                if client_idx in args.test_envs:
                    pass
                else:
                    val_acc = test(
                        args, model, val_loaders[client_idx], device)

                    val_acc_list[client_idx] = val_acc
                    print(' Site-{:d}| Val  Acc: {:.4f}'.format(
                        client_idx, val_acc), flush=True)
                    logrecord += ' Site-{:d}| Val  Acc: {:.4f}\n'.format(
                        client_idx, val_acc)



            test_acc_list = [0. for j in range(client_num)]
            for client_idx in range(client_num):
                if client_idx in args.test_envs:
                    test_acc = test(args, server_model,
                                    test_loaders[client_idx], device)
                    print(
                        ' Test site-{:d}| Test Acc: {:.4f}'.format(client_idx, test_acc))
                    logrecord += ' Test site-{:d}| Test Acc: {:.4f}'.format(
                        client_idx, test_acc)
                    test_acc_list[client_idx] = test_acc

            for model in models:
                if a_iter % 2 == 0:
                    model.get_global_prompts()




            if a_iter %2 == 0:
                mean_val_acc = 0
            else:
                mean_val_acc = np.mean(val_acc_list)
            print(f"valid mean accuracy: {mean_val_acc}")
            if mean_val_acc > np.mean(best_acc):
                for client_idx in range(client_num):
                    if client_idx in args.test_envs:
                        pass
                    else:
                        best_acc[client_idx] = val_acc_list[client_idx]
                        best_epoch = a_iter
                        best_changed = True

            if best_changed:
                finalrecord = finalrecord + str(a_iter) + ','
                for item in test_acc_list:
                    finalrecord = finalrecord + str(item) + ','
                best_changed = False
        end_time = time.time()
        print("One epoc takes time: {:.2f}s".format(end_time - start_time))
    print("saving model in root dir.")
    torch.save(server_model, 'Domainnet' + str(args.test_envs[0]) + '.pth')


    print('best epoch:%d\n' % (best_epoch))
    logrecord += '\n best epoch:%d\n' % (best_epoch)
    rec = finalrecord.split(',')[-5:-1]
    ts = ''
    for item in rec:
        ts += '%.4f ' % float(item)
    print('best test acc: ' + ts)
    logrecord += 'best test acc: ' + ts
    filename = 'results/Maple_deep_01/' + args.optimizers + '_' + args.dataset + '_lr1_' + str(args.lr1) + 'a_t_' + str(args.a_t)
    filename = filename + '_' + args.backbone_name
    os.makedirs(filename, exist_ok=True)
    output_txt = '/output_' + str(args.test_envs[0]) + '-' + str(args.iters) + '-' + str(args.wk_iters) + '.txt'
    with open(filename + output_txt, 'w') as f:
        f.write(finalrecord)
    log_txt = '/log_' + str(args.test_envs[0]) + '--' + \
              str(args.iters) + '--' + str(args.num_tarin_client_prompts) + '--' + str(args.num_tarin_adapters) + '.txt'
    with open(filename + log_txt, 'w') as f:
        f.write(logrecord)

