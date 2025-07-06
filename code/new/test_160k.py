# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math
from utils import load_network
from image_folder import CustomData160k_sat, CustomData160k_drone
import torch.nn.functional as F

#fp16
try:
    from apex.fp16_utils import *
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='./data/test',type=str, help='./test_data')
parser.add_argument('--name', default='three_view_long_share_d0.75_256_s1_google', type=str, help='save model path')
parser.add_argument('--pool', default='avg', type=str, help='avg|max')
parser.add_argument('--batchsize', default=128, type=int, help='batchsize')
parser.add_argument('--PCB', action='store_true', help='use PCB' )
parser.add_argument('--h', default=256, type=int, help='height')
parser.add_argument('--w', default=256, type=int, help='width')
parser.add_argument('--views', default=2, type=int, help='views')
parser.add_argument('--pad', default=0, type=int, help='padding')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--LPN', action='store_true', help='use LPN' )
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--fp16', action='store_true', help='use fp16.' )
parser.add_argument('--scale_test', action='store_true', help='scale test' )
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument('--query_name', default='query_street_name.txt', type=str,help='load query image')
opt = parser.parse_args()
###load config###
# load the training config
config_path = os.path.join('./model',opt.name,'opts.yaml')
with open(config_path, 'r') as stream:
        config = yaml.load(stream,Loader=yaml.FullLoader)
opt.fp16 = config['fp16']
opt.use_dense = config['use_dense']
opt.use_NAS = config['use_NAS']
opt.stride = config['stride']
opt.views = config['views']
opt.LPN = False
opt.block = 0
scale_test = opt.scale_test
if 'h' in config:
    opt.h = config['h']
    opt.w = config['w']
print('------------------------------',opt.h)
if 'nclasses' in config: # tp compatible with old config files
    opt.nclasses = config['nclasses']
else: 
    opt.nclasses = 729 

str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir
query_name = opt.query_name
gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

print('We use the scale: %s'%opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
data_transforms = transforms.Compose([
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#像素点平移动的transforms
transform_move_list = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


if opt.LPN:
    data_transforms = transforms.Compose([
        # transforms.Resize((384,192), interpolation=3),
        transforms.Resize((opt.h,opt.w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])


data_dir = test_dir

image_datasets = {}
image_datasets['gallery_satellite'] = CustomData160k_sat(os.path.join(data_dir, 'workshop_gallery_satellite'), data_transforms)
image_datasets['query_street'] = CustomData160k_drone( os.path.join(data_dir,'workshop_query_street') ,data_transforms, query_name = query_name)
print(image_datasets.keys())


dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                shuffle=False, num_workers=16) for x in
                ['gallery_satellite','query_street']}

use_gpu = torch.cuda.is_available()

######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#

# ===== 在所有 import 之后，定义 re_ranking 函数 =====
import numpy as np

def re_ranking(dist_q_g, dist_q_q, dist_g_g, k1=20, k2=6, lambda_value=0.3):
    all_num = dist_q_q.shape[0] + dist_g_g.shape[0]
    original_dist = np.zeros((all_num, all_num), dtype=np.float32)
    # 构建大矩阵
    original_dist[:dist_q_q.shape[0], :dist_q_q.shape[1]] = dist_q_q
    original_dist[:dist_q_g.shape[0], dist_q_q.shape[1]:] = dist_q_g
    original_dist[dist_q_q.shape[0]:, :dist_q_q.shape[1]] = dist_q_g.T
    original_dist[dist_q_q.shape[0]:, dist_q_q.shape[1]:] = dist_g_g

    V = np.zeros_like(original_dist, dtype=np.float32)
    # 计算 k-reciprocal 邻居
    for i in range(all_num):
        forward_k = np.argsort(original_dist[i])[:k1+1]
        backward_k = [j for j in forward_k
                      if i in np.argsort(original_dist[j])[:k1+1]]
        k_recip = np.array(backward_k)
        weight = np.exp(-original_dist[i, k_recip])
        V[i, k_recip] = weight / np.sum(weight)

    # 局部 QE
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i] = np.mean(V[np.argsort(original_dist[i])[:k2]], axis=0)
        V = V_qe

    # Jaccard 距离
    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)
    invIndex = [np.where(V[:, i] != 0)[0] for i in range(all_num)]
    for i in range(all_num):
        temp_min = np.zeros((1, all_num), dtype=np.float32)
        nz = np.where(V[i] != 0)[0]
        for j in nz:
            temp_min[0, invIndex[j]] += np.minimum(
                V[i, j], V[invIndex[j], j]
            )
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    # 最终融合
    final_dist = (1 - lambda_value) * jaccard_dist + lambda_value * original_dist
    # 只返回 query-vs-gallery 部分
    return final_dist[:dist_q_g.shape[0], dist_q_q.shape[1]:]

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def which_view(name):
    if 'satellite' in name:
        return 1
    elif 'street' in name:
        return 2
    elif 'drone' in name:
        return 3
    else:
        print('unknown view')
    return -1

def extract_feature(model,dataloaders, view_index = 1):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        print(count)
        ff = torch.FloatTensor(n,512).zero_().cuda()
        # if opt.LPN:
        #     # ff = torch.FloatTensor(n,2048,6).zero_().cuda()
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bilinear', align_corners=False)
                if view_index == 1:
                    outputs,_, _ = model(input_img, None,None)
                elif view_index == 2:
                    _, outputs,_ = model(None, input_img, None)
                # outputs, outputs_se = model(input_img)

                ff += outputs
        # norm feature
        if opt.LPN:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(10)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff.data.cpu()), 0)
    return features

def get_SatId_160k(img_path):
    labels = []
    paths = []
    for path,v in img_path:
        labels.append(v)
        paths.append(path)
    return labels, paths

def get_result_rank10(qf,gf,gl):
    query = qf.view(-1,1)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    index = np.argsort(score)
    index = index[::-1]
    rank10_index = index[0:10]
    result_rank10 = gl[rank10_index]
    return result_rank10

######################################################################
# Load Collected data Trained model
print('-------test-----------')

model, _, epoch = load_network(opt.name, opt)
if opt.LPN:
    print('use LPN')
    # model = three_view_net_test(model)
    for i in range(opt.block):
        cls_name = 'classifier'+str(i)
        c = getattr(model, cls_name)
        c.classifier = nn.Sequential()
else:
    model.classifier.classifier = nn.Sequential()
model = model.eval()
if use_gpu:
    model = model.cuda()

# Extract feature
since = time.time()

query_name = 'query_street'    #1
gallery_name = 'gallery_satellite'   #1

which_gallery = which_view(gallery_name)
which_query = which_view(query_name)

gallery_path = image_datasets[gallery_name].imgs
gallery_label, gallery_path  = get_SatId_160k(gallery_path)

print('%d -> %d:'%(which_query, which_gallery))

if __name__ == "__main__":
    with torch.no_grad():
        print('-------------------extract query feature----------------------')
        query_feature = extract_feature(model, dataloaders[query_name], which_query)
        print('-------------------extract gallery feature----------------------')
        gallery_feature = extract_feature(model, dataloaders[gallery_name], which_gallery)
        print('--------------------------ending extract-------------------------------')

    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # -- 1) Move to GPU --
    Q = query_feature.cuda()
    G = gallery_feature.cuda()

    # -- 2) PCA + Whitening --
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize

    # 从 GPU tensor 转回 CPU numpy
    Q_np = Q.cpu().numpy()
    G_np = G.cpu().numpy()
    all_feats = np.vstack([Q_np, G_np])     # shape = (Nq+Ng, D)

    # 白化并降到 512 维
    pca = PCA(n_components=512, whiten=True)
    all_white = pca.fit_transform(all_feats)

    # 拆回 Q 和 G
    Q_white = all_white[:Q_np.shape[0], :]
    G_white = all_white[Q_np.shape[0]:, :]

    # L2 归一化
    Q_np = normalize(Q_white, axis=1)
    G_np = normalize(G_white, axis=1)

    # 转回 GPU
    Q = torch.from_numpy(Q_np).cuda()
    G = torch.from_numpy(G_np).cuda()

    # -- 3) k-reciprocal 重排 --
    dist_q_g = 1 - torch.mm(Q, G.t()).cpu().numpy()
    dist_q_q = 1 - torch.mm(Q, Q.t()).cpu().numpy()
    dist_g_g = 1 - torch.mm(G, G.t()).cpu().numpy()
    rerank_dist = re_ranking(dist_q_g, dist_q_q, dist_g_g,
                             k1=20, k2=6, lambda_value=0.3)

    # -- 4) Average Query Expansion (AQE) --
    K = 5
    for i in range(Q.size(0)):
        topk = np.argsort(rerank_dist[i])[:K]
        agg = torch.from_numpy(G_np[topk]).sum(0).cuda()
        Q[i] = F.normalize(Q[i] + agg, dim=0)

    # -- 5) 生成 final_score (这里直接用 rerank 距离) --
    final_score = rerank_dist  # shape = (num_query, num_gallery)

    # -- 6) 排序、写文件并统计 Recall@1/5/10 --
    Nq = final_score.shape[0]
    R1 = R5 = R10 = 0
    gallery_labels = np.array(gallery_label)
    save_filename = 'answer.txt'
    if os.path.isfile(save_filename):
        os.remove(save_filename)
    with open(save_filename, 'w') as f:
        for i in range(Nq):
            idx = np.argsort(final_score[i])    # 距离越小越相似
            top10 = idx[:10]
            # 写入 top-10
            f.write('\t'.join(map(str, gallery_labels[top10])) + '\n')
            # 统计 Recall
            gt = image_datasets[query_name].imgs[i][1]
            if top10[0] == gt:
                R1 += 1
            if gt in top10[:5]:
                R5 += 1
            if gt in top10:
                R10 += 1

    print(f"Recall@1:  {R1/Nq*100:.4f}%")
    print(f"Recall@5:  {R5/Nq*100:.4f}%")
    print(f"Recall@10: {R10/Nq*100:.4f}%")
    print("You need to compress the file as *.zip before submission.")
