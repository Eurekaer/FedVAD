from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import time
import numpy as np
import random
from configs import build_config
from utils import setup_seed,Visualizer
from log import get_logger
from model import XModel
from dataset import *
from train import train_func,public_train_func
from test import test_func
import argparse
import copy
import os
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from cluster_utils import *
from fusion import *
from distill import *

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
log_dir = ''
writer = SummaryWriter(log_dir)
rounds = 102
num_clients = 7
device = torch.device("cuda")
seed = 42
torch.manual_seed(seed)

def add_noise_to_state_dict(state_dict, noise_type, sigma, device):
    new_state_dict = {}
    for name, param in state_dict.items():
        if noise_type == 1:
            # 拉普拉斯噪声
            noise = torch.tensor(np.random.laplace(0, sigma, param.size()), device=device)
        elif noise_type == 2:
            # 高斯噪声
            noise = torch.randn(param.size(), device=device) * sigma
        else:
            # 不添加噪声
            noise = torch.zeros(param.size(), device=device)
        
        new_state_dict[name] = param + noise
    return new_state_dict

def average_weights(weight_list):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    avg_weights = {}
    for key in weight_list[0].keys():
        tensor_list = [client_weights[key].float().to(device) for client_weights in weight_list]  
        avg_weights[key] = torch.stack(tensor_list).mean(dim=0)
    return avg_weights
def load_checkpoint(model, ckpt_path, logger):
    if os.path.isfile(ckpt_path):
        logger.info('loading pretrained checkpoint from {}.'.format(ckpt_path))
        weight_dict = torch.load(ckpt_path)
        model_dict = model.state_dict()
        for name, param in weight_dict.items():
            if 'module' in name:
                name = '.'.join(name.split('.')[1:])
            if name in model_dict:
                if param.size() == model_dict[name].size():
                    model_dict[name].copy_(param)
                else:
                    logger.info('{} size mismatch: load {} given {}'.format(
                        name, param.size(), model_dict[name].size()))
            else:
                logger.info('{} not found in model dict.'.format(name))
    else:
        logger.info('Not found pretrained checkpoint file.')

def train(model, train_loader, test_loader, client_idx,gt, logger):
    if not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)

    criterion = torch.nn.BCELoss()
    criterion2 = torch.nn.KLDivLoss(reduction='batchmean')
    dl_loss_function = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=0)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_auc = 0.0
    auc_far = 0.0

    st = time.time()

    best_epoch = 0

    for epoch in range(cfg.max_epoch):
        print("epoch = {}".format(epoch))
        loss1, loss2 = train_func(train_loader, model, optimizer, criterion, criterion2,dl_loss_function,client_idx, cfg.lamda)

        scheduler.step()

        auc, far = test_func(test_loader, model, gt, cfg.dataset)
        if auc >= best_auc:
            best_auc = auc
            auc_far = far
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())

        logger.info('[Epoch:{}/{}]: loss1:{:.4f} loss2:{:.4f} | AUC:{:.4f} FAR:{:.5f}'.format(
            epoch + 1, cfg.max_epoch, loss1, loss2, auc, far))

    print("best_epoch = {}".format(best_epoch))
    print("best_auc = {}".format(best_auc))

    time_elapsed = time.time() - st 
    model.load_state_dict(best_model_wts)
    logger.info('Training completes in {:.0f}m {:.0f}s | best {}:{:.4f} FAR:{:.5f}\n'.
                format(time_elapsed // 60, time_elapsed % 60, cfg.metrics, best_auc, auc_far))
    return best_model_wts,best_auc

def double_cluster(local_weights):
    num_clusters = 3 
    feature_size = 1207347   # model parameter size
    bert_feature_size = 768  # BERT feature dim
    bert_features = []
    for i in range(1,8):
        path = 'each client object target feature path.'
        fea = np.load(path) 
        bert_features.append(fea)
    params_matrix = []
    for w in local_weights:
        params_vector = torch.cat([v.flatten() for v in w.values()])
        params_matrix.append(params_vector.detach().cpu().numpy())
    params_matrix = np.array(params_matrix)
    # Randomly initialize cluster centers
    cluster_centers_w = torch.randn(num_clusters, feature_size)
    cluster_centers_o = torch.randn(num_clusters, bert_feature_size)

    alpha = 0.4
    beta = 0.6

    # iterative process
    num_iterations = 10 
    for iteration in range(num_iterations):
        # Step 1: Assign clusters
        r_ik = cluster_assignment_update(params_matrix, bert_features, cluster_centers_w, cluster_centers_o, alpha, beta)
        
        # Step 2: Update cluster centers
        cluster_centers_w, cluster_centers_o = update_cluster_centers(r_ik, params_matrix, bert_features)

        # Step 3: Retrieve the cluster indices for each data point
        cluster_indices = torch.argmax(r_ik, dim=1)

        print("iteration process is {}".format(iteration))
    print("Final cluster centers for weights:\n", cluster_centers_w)
    print("Final cluster centers for distributions:\n", cluster_centers_o)
    print("Cluster indices for each data point:\n", cluster_indices)
    print("cluster end !!!")
    c = []
    c0 = []
    c1 = []
    c2 = []
    c3 = []
    c4 = []
    classes = cluster_indices
    for i in range(0,len(classes)):
        if classes[i] == 0:
            c0.append(i)
        if classes[i] == 1:
            c1.append(i)
        if classes[i] == 2:
            c2.append(i)            
        if classes[i] == 3:
            c3.append(i)
        if classes[i] == 4:
            c4.append(i)
    if len(c0) != 0:
        c.append(c0)
    if len(c1) != 0:
        c.append(c1)
    if len(c2) != 0:
        c.append(c2)
    if len(c3) != 0:
        c.append(c3)
    if len(c4) != 0:
        c.append(c4)
    return c

def main(cfg,global_model):
    global_model = global_model
    print(global_model)
    logger = get_logger(cfg.logs_dir)
    setup_seed(cfg.seed)
    for round in range(rounds):
        local_weights = []
        auc_list = []
        print(f"Round {round + 1} of {rounds}")
        flag = False
        for client in range(1,num_clients+1):
            print("client = {}".format(client))
            if cfg.dataset == 'shanghaiTech':
                cfg.train_list = 'each client train list path.'
                cfg.test_list = 'each client test list path.'
                cfg.gt = 'each client gt list path.'
            elif cfg.dataset == 'ucf-crime':
                cfg.train_list = 'each client train list path.'
                cfg.test_list = 'each client test list path.'
                cfg.gt = 'each client gt list path.'
            else:
                cfg.train_list = 'each client train list path.'
                cfg.test_list = 'each client test list path.'
                cfg.gt = 'each client gt list path.'  
            print("client = {},dataset = {},train_list = {}".format(client,cfg.dataset,cfg.train_list)) 
            logger.info('Config:{}'.format(cfg.__dict__))
            if cfg.dataset == 'ucf-crime':
                train_data = UCFDataset(cfg, test_mode=False)
                test_data = UCFDataset(cfg, test_mode=True)
                flag = True
            elif cfg.dataset == 'xd-violence':
                train_data = XDataset(cfg, test_mode=False)
                test_data = XDataset(cfg, test_mode=True)
            elif cfg.dataset == 'shanghaiTech':
                train_data = SHDataset(cfg, test_mode=False)
                test_data = SHDataset(cfg, test_mode=True)
            else:
                raise RuntimeError("Do not support this dataset!")
            print("train_data_length = {}".format(len(train_data)))
            print("test_data_length = {}".format(len(test_data)))
            print(cfg.train_list)
            train_loader = DataLoader(train_data, batch_size=cfg.train_bs, shuffle=True,
                                    num_workers=cfg.workers, pin_memory=True)
            print("train_loader!!!")
            test_loader = DataLoader(test_data, batch_size=cfg.test_bs, shuffle=False,
                                    num_workers=cfg.workers, pin_memory=True)

            local_model = XModel(cfg)
            """ ================================ Add distillation operation ===================================="""
            if round != 0:
                local_model.load_state_dict(global_group_weights_dicts[client-1])
            else:
                local_model.load_state_dict(global_model.state_dict())
            """ ================================ Not add distillation operation ===================================="""
            gt = np.load(cfg.gt)
            device = torch.device("cuda")
            local_model = local_model.to(device)

            param = sum(p.numel() for p in local_model.parameters())
            logger.info('total params:{:.4f}M'.format(param / (1000 ** 2)))
            if args.mode == 'train':
                logger.info('Training Mode')
                client_idx = client
                best_model_wts,best_auc = train(local_model, train_loader, test_loader, client_idx,gt, logger)
            noise_type = 2
            sigma = 0.01
            # print(best_model_wts == local_model.parameters())
            update_bese_model_parameters = add_noise_to_state_dict(best_model_wts,noise_type,sigma,device)
            local_weights.append(update_bese_model_parameters)
            # auc_list.append(best_auc)
            # local_weights.append(best_model_wts)
            auc_list.append(best_auc)
        total_auc = sum(auc_list) / len(auc_list)

        writer.add_scalar("AUC", total_auc,round)
    
        global_group_weights_dicts = {}
        if round % 5 == 0:
            global_weights = average_weights(local_weights)
            for i in range(0,13):
                global_group_weights_dicts[i] = global_weights
        else: 
            tags = double_cluster(local_weights)       
            global_group_weights = []
            print(len(tags))
            for tag in tags:
                group_k = []
                print(tag)
                public_train_list_path = get_public_aug_train_list(tag)
                for k in range(0,len(tag)):
                    group_k.append(local_weights[tag[k]])
                global_weights = average_weights(group_k)
                global_model.load_state_dict(global_weights)
                global_model_weights_k = distill(global_model,global_weights,cfg,public_train_list_path)
                global_group_weights.append(global_model_weights_k)
                for k in tag:
                    global_group_weights_dicts[k] = global_model_weights_k

def distill(global_model,global_weights,args,public_train_list_path):

    global_model.load_state_dict(global_weights)
    visual_model_checkpoints = 'PEL4VAD pretrain visual model checkpoints path.'
    args.dataset = 'shanghaiTech'
    cfg = build_config(args.dataset)
    visual_model = XModel(cfg)
    visual_model.to("cuda:0")
    checkpoint = torch.load(visual_model_checkpoints)
    visual_model.load_state_dict(checkpoint, strict=False)

    cfg.train_list = public_train_list_path
    train_data = SHDataset(cfg, test_mode=False)

    train_loader = DataLoader(train_data, batch_size=cfg.train_bs, shuffle=True,
                                    num_workers=cfg.workers, pin_memory=True)
    global_model_weights = public_train(global_model,visual_model,train_loader)

    return global_model_weights

def public_train(global_model,visual_model,train_loader):
    global_model.train()
    visual_model.eval()

    criterion = torch.nn.BCELoss()
    criterion2 = torch.nn.KLDivLoss(reduction='batchmean')
    dl_loss_function = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(global_model.parameters(), lr=cfg.lr)

    for epoch in range(1):
        print("epoch = {}".format(epoch))
        global_model_weights = public_train_func(train_loader, global_model,visual_model, optimizer, criterion, criterion2,dl_loss_function, cfg.lamda)
    return global_model_weights

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WeaklySupAnoDet')
    parser.add_argument('--dataset', default='ucf', help='anomaly video dataset')
    parser.add_argument('--mode', default='train', help='model status: (train or infer)')
    args = parser.parse_args()
    cfg = build_config(args.dataset)
    global_model = XModel(cfg)
    main(cfg,global_model)