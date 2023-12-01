import os

A,B1,B2,B4,B5,B6,G = [],[],[],[],[],[],[]

with open('/home/tjut_panruijie/Weakly_Method/PEL4VAD-DL-New-Text/list/xd/client1.list','r') as test:
    info = test.readlines()
    for i in info:
        i = i.strip()
        if 'label_A__0' in i:
            A.append(i)
        if 'label_B1' in i:
            B1.append(i)
        if 'label_B2' in i:
            B2.append(i)
        if 'label_B4' in i:
            B4.append(i)
        if 'label_B5' in i:
            B5.append(i)
        if 'label_B6' in i:
            B6.append(i)
        if 'label_G' in i:
            G.append(i)
print(len(A))
print(len(B1))
print(len(B2))
print(len(B4))
print(len(B5))
print(len(B6))
print(len(G))



# with open('/home/tjut_panruijie/Weakly_Method/PEL4VAD-DL-New-Text/list/xd/train_A.list','w') as c1:
#     for b1 in A:
#         c1.writelines(b1 + '\n')



# 将 A 中的数据随机划分为 6 份

import random

def split_list_to_chunks(lst, n):
    """将列表 lst 随机划分为n个尽可能等长的子列表"""
    random.seed(42)
    random.shuffle(lst)  # 随机打乱列表
    return [lst[i::n] for i in range(n)]

chunks = split_list_to_chunks(A, 6) 
print(len(chunks[0]))

# with open('/home/tjut_panruijie/Weakly_Method/PEL4VAD-DL-New-Text/list/xd/client7.list','a') as a:
#     for ch in chunks[5]:

#         a.writelines(ch + '\n')

#         for k in range(1,5):
#             na = ch[:-5] + str(k) + '.npy'
#             a.writelines(na + '\n')



# base_path = '/home/tjut_panruijie/Weakly_Method/PEL4VAD-DL-New-Text/list/xd/xd-gt/'

# folders = os.listdir(base_path)

import numpy as np

# s = 0

# for f in folders:
#     t_p = base_path + f
#     s += len(np.load(t_p))

# print(s)

# print(len(np.load('/home/tjut_panruijie/Weakly_Method/PEL4VAD-DL-New-Text/list/xd/xd-gt.npy')))


# for i in range(1,7):
#     client_path = '/home/tjut_panruijie/Weakly_Method/PEL4VAD-DL-New-Text/list/xd/new_split/test_clients/clients/client' + str(i) + '.list'
#     gt_names = []
#     with open(client_path,'r') as cp:
#         test_list = cp.readlines()
#         for l in test_list:
#             l = l.strip()
#             if '__0' in l:
#                 gt_names.append(l)
#         # print(gt_names)

#         gt_base_path = '/home/tjut_panruijie/Weakly_Method/PEL4VAD-DL-New-Text/list/xd/xd-gt/'

#         gt_feature = np.zeros(0)
#         for gt in gt_names:
#             gt = gt.split('/')[1]
#             content = np.load(gt_base_path + gt)
#             gt_feature = np.concatenate((gt_feature,content))
#         print(len(gt_feature))
#         np.save('/home/tjut_panruijie/Weakly_Method/PEL4VAD-DL-New-Text/list/xd/new_split/test_gts/client' + str(i) + '.pkl',gt_feature)
    # break
def split_list_by_ratio(lst, ratio, seed=None):
    """将列表 lst 按照给定的比例 ratio 随机划分成两个子列表"""
    if seed is not None:
        random.seed(seed)  # 设置随机种子，确保结果可复现
    random.shuffle(lst)  # 随机打乱列表
    
    # 计算切分点
    split_point = int(len(lst) * ratio)
    
    # 划分列表
    return lst[:split_point], lst[split_point:]

for i in range(1,7):
    client_path = '/home/tjut_panruijie/Weakly_Method/PEL4VAD-DL-New-Text/list/xd/new_split/train_clients/client' + str(i) + '.list'
    gt_names = []
    with open(client_path,'r') as cp:
        train_list = cp.readlines()
        for l in train_list:
            l = l.strip()
            if '__0' in l:
                gt_names.append(l)
        # print(gt_names)
        print(len(gt_names))

        # 调用函数，将列表按照1:9的比例随机分成两个子列表
        smaller, larger = split_list_by_ratio(gt_names, 0.1, seed=42)

        # 输出结果
        print(f"Smaller list: {len(smaller)}")
        print(f"Larger list: {len(larger)}")


        new_train_client_path = '/home/tjut_panruijie/Weakly_Method/PEL4VAD-DL-New-Text/list/xd/new_split/new_train_clients/client'
        with open(new_train_client_path + str(i) + '.list','w') as nt: 
            for s in larger:
                nt.writelines(s + '\n')
                for k in range(1,5):
                    na = s[:-5] + str(k) + '.npy'
                    nt.writelines(na + '\n')
        public_train_client_path = '/home/tjut_panruijie/Weakly_Method/PEL4VAD-DL-New-Text/list/xd/new_split/public_train_clients/public_train'
        with open(public_train_client_path + '.list','a') as nt: 
            for s in smaller:
                nt.writelines(s + '\n')
                for k in range(1,5):
                    na = s[:-5] + str(k) + '.npy'
                    nt.writelines(na + '\n')
        # break


