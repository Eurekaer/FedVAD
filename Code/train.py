import torch
from loss import *
from utils import *
from bert_model import BERT_Arch
from transformers import AutoModel, BertTokenizerFast
import pickle as pk
from fusion import *

def train_func(dataloader, model, text_model_weights, optimizer, criterion, criterion2, dl_loss_function,client_idx, lamda=0):
    t_loss = []
    s_loss = []

    with torch.set_grad_enabled(True):
        model.train()
        for i, (v_input, t_input, label, multi_label, t_id) in enumerate(dataloader):
        # for i, (v_input, t_input, label, multi_label) in enumerate(dataloader):
            seq_len = torch.sum(torch.max(torch.abs(v_input), dim=2)[0] > 0, 1)
            v_input = v_input[:, :torch.max(seq_len), :]
            v_input = v_input.float().cuda(non_blocking=True)
            t_input = t_input.float().cuda(non_blocking=True)
            label = label.float().cuda(non_blocking=True)
            multi_label = multi_label.cuda(non_blocking=True)

            logits, v_feat = model(v_input, seq_len)
            
            # Prompt-Enhanced Learning
            logit_scale = model.logit_scale.exp()
            video_feat, token_feat, video_labels = get_cas(v_feat, t_input, logits, multi_label)

            v2t_logits, v2v_logits = create_logits(video_feat, token_feat, logit_scale)
            ground_truth = torch.tensor(gen_label(video_labels), dtype=v_feat.dtype).cuda()


            loss2 = KLV_loss(v2t_logits, ground_truth, criterion2)

            loss1 = CLAS2(logits, label, seq_len, criterion)

            loss = loss1 + lamda * loss2 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_loss.append(loss1)
            s_loss.append(loss2)

    return sum(t_loss) / len(t_loss), sum(s_loss) / len(s_loss)


def public_train_func(dataloader, global_model,visual_model, optimizer, criterion, criterion2,dl_loss_function, lamda=0):
    t_loss = []
    s_loss = []
    mse_criterion = nn.MSELoss()    
    with torch.set_grad_enabled(True):
        global_model.train().cuda()
        visual_model.eval().cuda()
        feature_dim = 512
        num_heads = 8
        feature_fusion_model = FeatureFusionMHSA(feature_dim, num_heads).to("cuda")
        for i, (v_input, t_input, label, multi_label, t_id) in enumerate(dataloader):
            seq_len = torch.sum(torch.max(torch.abs(v_input), dim=2)[0] > 0, 1)
            v_input = v_input[:, :torch.max(seq_len), :]
            v_input = v_input.float().cuda(non_blocking=True)
            t_input = t_input.float().cuda(non_blocking=True)
            label = label.float().cuda(non_blocking=True)
            multi_label = multi_label.cuda(non_blocking=True)

            logits, v_feat = global_model(v_input, seq_len)
            logits2, v_feat2 = visual_model(v_input, seq_len)

            # t_feature = get_xd_text_feature(t_id) 
            t_feature = get_sh_text_feature(t_id) 
            t_feature_expanded = t_feature.unsqueeze(2).cuda(non_blocking=True)  # shape: [128, 512, 1]
            t_feature_expanded = t_feature_expanded.expand(-1, -1, v_feat.size(-1))
        

            fused_feature = feature_fusion_model(v_feat,t_feature_expanded)

            mse_loss = mse_criterion(v_feat,fused_feature)


            # Prompt-Enhanced Learning
            logit_scale = global_model.logit_scale.exp()
            video_feat, token_feat, video_labels = get_cas(v_feat, t_input, logits, multi_label)

            v2t_logits, v2v_logits = create_logits(video_feat, token_feat, logit_scale)
            ground_truth = torch.tensor(gen_label(video_labels), dtype=v_feat.dtype).cuda()


            loss2 = KLV_loss(v2t_logits, ground_truth, criterion2)

            loss1 = CLAS2(logits, label, seq_len, criterion)

            loss = loss1 + lamda * loss2 + 0.7 * mse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_loss.append(loss1)
            s_loss.append(loss2)

    return global_model.state_dict()

def get_text_id_based_name(name):
    import csv
    data_dict = {}
    csv_file =  name # 替换为你的 CSV 文件路径
    with open(csv_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            video_name = row['name']
            sentence = row['id']
            data_dict[sentence] = video_name      
    return data_dict  
def get_sh_text_feature(t_id):
    fe = np.zeros((len(t_id), 512))
    linear_layer = create_linear_layer(768, 512)

    with open('all sh_text_feature path','rb') as tf:
        infos = pk.load(tf)
    ids = get_text_id_based_name('Path to the id of the text corresponding to the video in the public dataset SH.')
    for i, id in  enumerate(t_id):
        name = ids[str(id)]

        feature = infos[name].reshape(768)
        f = linear_layer(feature).reshape(1, 512)
        fe[i] = f.cpu().detach().numpy()
    fe_tensor = torch.from_numpy(fe).float() 

    return fe_tensor

import torch.nn as nn

def create_linear_layer(input_dim, output_dim):
    return nn.Linear(input_dim, output_dim)

# 使用函数
get_sh_text_feature([10])

