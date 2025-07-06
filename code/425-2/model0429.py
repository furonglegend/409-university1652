import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.models import convnext_base
from pytorch_metric_learning.models import NetVLAD
import timm


######################################################################

# 在文件头部引入
import math
# model.py
# 1) 定义 CrossBranchFusion
class CrossBranchFusion(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=feat_dim,
                                          num_heads=8,
                                          batch_first=True)
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, feats):
        # feats: list of [B, Dp]
        x = torch.stack(feats, dim=1)        # [B, 4, Dp]
        y, _ = self.attn(x, x, x)            # self‐attention
        y = self.norm(x + y)                 # residual + LayerNorm
        return y.mean(dim=1)                 # [B, Dp]

        
class MultiBackboneEnsemble(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, pool='avg'):
        super().__init__()

        # --- 1) 原有骨干 + 新增 Transformer 系列 ---
        self.backbones = nn.ModuleDict({
            'shuffle_vit'     : ft_net_ShuffleViT(class_num, droprate, stride, pool=pool),
            'convnext'        : None,
            'resnet50_conv'   : nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2]),
            'resnet50_pool'   : nn.AdaptiveAvgPool2d((1,1)),
            'vit_large'       : timm.create_model('vit_large_patch16_384', pretrained=True, num_classes=0),
            'efficientnet_b7' : timm.create_model('efficientnet_b7', pretrained=True, num_classes=0),
            'resnest101'      : timm.create_model('resnest101e', pretrained=True, num_classes=0),
            'swin_base'       : timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0),
            # 新增：
            'swin_large'      : timm.create_model('swin_large_patch4_window12_384', pretrained=True, num_classes=0),
            'mvit_base'       : timm.create_model('mvit_base_16x4', pretrained=True, num_classes=0),
            'beit_base'       : timm.create_model('beit_base_patch16_224', pretrained=True, num_classes=0),
            'deit_base'       : timm.create_model('deit_base_distilled_patch16_224', pretrained=True, num_classes=0),
        })
        # load ConvNeXt
        convnext = convnext_base(pretrained=True)
        convnext_dim = convnext.classifier[1].in_features
        self.backbones['convnext'] = nn.Sequential(*list(convnext.children())[:-1])

        # --- 2) Region‐level 聚合器 NetVLAD (不变) ---
        self.netvlad = NetVLAD(num_clusters=32, dim=2048)

        # --- 3) 计算各支路维度 & Dp ---
        shuffle_dim     = self.backbones['shuffle_vit'].in_channels
        resnet_dim      = 2048
        vit_dim         = self.backbones['vit_large'].embed_dim
        effb7_dim       = self.backbones['efficientnet_b7'].num_features
        resnest_dim     = self.backbones['resnest101'].num_features
        swin_base_dim   = self.backbones['swin_base'].num_features
        swin_large_dim  = self.backbones['swin_large'].num_features
        mvit_dim        = self.backbones['mvit_base'].num_features
        beit_dim        = self.backbones['beit_base'].num_features
        deit_dim        = self.backbones['deit_base'].num_features
        rmac_dim        = resnet_dim * 32

        total_dim = (
            shuffle_dim + convnext_dim + resnet_dim +
            vit_dim + effb7_dim + resnest_dim +
            swin_base_dim + swin_large_dim + mvit_dim +
            beit_dim + deit_dim + rmac_dim
        )
        Dp = total_dim

        # --- 4) 映射到 Dp ---
        self.fc_map = nn.ModuleDict({
            'shuffle_vit'     : nn.Linear(shuffle_dim,   Dp),
            'convnext'        : nn.Linear(convnext_dim,  Dp),
            'resnet50'        : nn.Linear(resnet_dim,    Dp),
            'vit_large'       : nn.Linear(vit_dim,       Dp),
            'efficientnet_b7' : nn.Linear(effb7_dim,     Dp),
            'resnest101'      : nn.Linear(resnest_dim,   Dp),
            'swin_base'       : nn.Linear(swin_base_dim, Dp),
            'swin_large'      : nn.Linear(swin_large_dim,Dp),
            'mvit_base'       : nn.Linear(mvit_dim,      Dp),
            'beit_base'       : nn.Linear(beit_dim,      Dp),
            'deit_base'       : nn.Linear(deit_dim,      Dp),
            'rmac'            : nn.Linear(rmac_dim,      Dp),
        })

        # --- 5) 融合 & 投影（不变） ---
        self.fusion = nn.Sequential(
            nn.LayerNorm(Dp),
            GeM(dim=Dp),
            CrossBranchFusion(feat_dim=Dp),
            nn.Linear(Dp, Dp), nn.ReLU(inplace=True), nn.Dropout(p=droprate),
        )
        self.proj = nn.Sequential(
            nn.Linear(Dp, int(Dp * 0.75)), nn.ReLU(inplace=True), nn.Dropout(p=droprate),
            nn.Linear(int(Dp * 0.75), total_dim // 2), nn.BatchNorm1d(total_dim // 2),
        )

    def forward(self, x):
        B = x.size(0)
        # — 全局分支 —
        f1 = self.backbones['shuffle_vit'](x)
        f2 = F.normalize(self.backbones['convnext'](x).view(B, -1),    p=2, dim=1)
        conv_map = self.backbones['resnet50_conv'](x)
        f3 = F.normalize(self.backbones['resnet50_pool'](conv_map).view(B, -1), p=2, dim=1)
        v  = self.backbones['vit_large'].forward_features(x)
        cls= F.normalize(v[:,0], p=2, dim=1)
        f4 = F.normalize(self.backbones['efficientnet_b7'](x).view(B, -1), p=2, dim=1)
        f5 = F.normalize(self.backbones['resnest101'](x).view(B, -1),   p=2, dim=1)
        f6 = F.normalize(self.backbones['swin_base'](x).view(B, -1),    p=2, dim=1)
        # — Swin-Large / MViT / BEiT / DeiT —
        f7  = F.normalize(self.backbones['swin_large'](x).view(B, -1), p=2, dim=1)
        f8  = F.normalize(self.backbones['mvit_base'](x).view(B, -1),  p=2, dim=1)
        f9  = F.normalize(self.backbones['beit_base'](x).view(B, -1),  p=2, dim=1)
        f10 = F.normalize(self.backbones['deit_base'](x).view(B, -1),  p=2, dim=1)

        # — 局部特征分支: NetVLAD —
        feat_flat = conv_map.view(B, 2048, -1)
        rmac_feats= self.netvlad(feat_flat)
        rmac_feats= F.normalize(rmac_feats, dim=1)

        # — 映射 & 融合 —
        mapped = [
            self.fc_map['shuffle_vit'](f1),
            self.fc_map['convnext'](f2),
            self.fc_map['resnet50'](f3),
            self.fc_map['vit_large'](cls),
            self.fc_map['efficientnet_b7'](f4),
            self.fc_map['resnest101'](f5),
            self.fc_map['swin_base'](f6),
            self.fc_map['swin_large'](f7),
            self.fc_map['mvit_base'](f8),
            self.fc_map['beit_base'](f9),
            self.fc_map['deit_base'](f10),
            self.fc_map['rmac'](rmac_feats),
        ]
        fused = self.fusion(mapped)
        return self.proj(fused)
        
class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, 2)
        neg_dist = F.pairwise_distance(anchor, negative, 2)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

# MultiScalePoolViT 使用 ShuffleNet 和 ViT 的结合
# MultiScalePoolShuffleViT 使用 ShuffleNet 和 ViT 的结合
class MultiScalePoolShuffleViT(nn.Module):
    def __init__(self, backbone_shufflenet, vit_backbone, dropout_rate=0.2):
        super().__init__()
        self.shufflenet = backbone_shufflenet
        self.vit        = vit_backbone

        # ViT 输出的 embed_dim
        dim = self.vit.embed_dim

        # GeM 池化
        self.gem     = GeM(dim=dim)
        # 可调 dropout
        self.dropout = nn.Dropout(dropout_rate)
        # MLP 用于动态计算融合权重 alpha
        self.alpha_fc = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, x):
        # 1. ShuffleNet 特征提取
        x = self.shufflenet(x)                      # [B, C]
        
        # 2. ViT Patch Embedding + Transformer
        x = self.vit.patch_embed(x)                 # [B, N, C]
        B = x.size(0)
        cls_tokens = self.vit.cls_token.expand(B, -1, -1)  
        x = torch.cat((cls_tokens, x), dim=1)       # [B, 1+N, C]
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        for blk in self.vit.blocks:
            x = blk(x)

        # 3. 分离出 [CLS] token 和 patch 特征
        class_tok = x[:, 0]                         # [B, C]
        patches   = x[:, 1:].transpose(1, 2)        # [B, C, N]
        N = patches.size(-1)
        H = W = int(math.sqrt(N))
        patches   = patches.view(B, dim, H, W)      # [B, C, H, W]

        # 4. GeM 池化
        ppool = self.gem(patches)                   # [B, C]

        # 5. 动态融合
        alpha = torch.sigmoid(self.alpha_fc(torch.cat([class_tok, ppool], dim=1)))  # [B,1]
        c     = class_tok * alpha                   # [B, C]
        p     = ppool * (1.0 - alpha)               # [B, C]
        feat  = torch.cat([c, p], dim=1)            # [B, 2C]

        # 6. Dropout + L2 归一化
        feat = self.dropout(feat)
        return F.normalize(feat, dim=1)             # [B, 2C]
        
# 轻量级 ShuffleNet
class ft_net_ShuffleViT(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net_ShuffleViT, self).__init__()
        
        # 使用 ShuffleNet 作为 Backbone
        shufflenet = timm.create_model('shufflenet_v2_x0_5', pretrained=True)
        shufflenet.fc = nn.Identity()  # 去掉 ShuffleNet 的分类层
        
        # 使用 ViT 作为 Transformer Backbone
        vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        vit.head = nn.Identity()
        
        # 组合 ShuffleNet 和 ViT
        self.model = MultiScalePoolShuffleViT(shufflenet, vit)
        self.in_channels = vit.embed_dim * 2  # ViT 输出的特征维度

        self.pool = pool
        if pool == 'gem':
            self.model.gem2 = GeM(dim=self.in_channels)

        self.se = nn.Identity()

    def forward(self, x):
        feat = self.model(x)
        return feat

class MultiScalePoolViT(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.vit = backbone
        self.gem = GeM(dim=self.vit.embed_dim)
        self.dropout = nn.Dropout(0.2)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 可学习的融合参数
        self.alpha_fc = nn.Sequential(
            nn.Linear(self.vit.embed_dim * 2, self.vit.embed_dim),  # 用于动态调整融合权重
            nn.ReLU(inplace=True),
            nn.Linear(self.vit.embed_dim, 1),
        )

    def forward(self, x):
        x = self.vit.patch_embed(x)  # [B, N, C]
        B = x.size(0)
        cls_tokens = self.vit.cls_token.expand(B, -1, -1)  # [B,1,C]
        x = torch.cat((cls_tokens, x), dim=1)  # [B,1+N,C]
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        for blk in self.vit.blocks:
            x = blk(x)

        class_tok = x[:, 0]  # [B, C]
        patches = x[:, 1:].transpose(1, 2)
        N = patches.size(-1)
        H = W = int(math.sqrt(N))
        patches = patches.view(B, self.vit.embed_dim, H, W)

        ppool = self.gem(patches)  # [B, C]

        # 动态调整 alpha 的值
        alpha_weight = torch.sigmoid(self.alpha_fc(torch.cat([class_tok, ppool], dim=1)))
        c = class_tok * alpha_weight
        p = ppool * (1.0 - alpha_weight)
        feat = torch.cat([c, p], dim=1)  # [B, 2C]
        feat = self.dropout(feat)
        return F.normalize(feat, dim=1)


class ArcMarginProduct(nn.Module):
    """ArcFace: https://arxiv.org/abs/1801.07698"""
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m

    def forward(self, x, labels):
        # x: [B, C], labels: [B]
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.clamp(cosine**2, 0, 1))
        phi = cosine * math.cos(self.m) - sine * math.sin(self.m)
        # 对于非该类样本，保持原 cosine
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1,1), 1.0)
        logits = self.s * (one_hot * phi + (1.0-one_hot) * cosine)
        return logits

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class GeM(nn.Module):
    def __init__(self, dim=2048, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(dim) * p, requires_grad=True)  # initial p
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        x = torch.transpose(x, 1, -1)
        x = x.clamp(min=eps).pow(p)
        x = torch.transpose(x, 1, -1)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        x = x.view(x.size(0), x.size(1))
        x = x.pow(1. / p)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ',' + 'dim='+str(self.dim)+')'

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

def fix_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x, f
        else:
            x = self.classifier(x)
            return x



class ft_net_VGG16(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net_VGG16, self).__init__()
        model_ft = models.vgg16_bn(pretrained=True)
        # 设置全局池化方式
        self.pool = pool
        if pool == 'avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
        elif pool == 'avg':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
        elif pool == 'max':
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
        elif pool == 'gem':
            model_ft.gem2 = GeM(dim=512)

        self.model = model_ft
        # 在特征提取后（即全局池化之前）插入 SE 模块，输入通道数为 512
        self.se = SEBlock(channel=512)

        if init_model is not None:
            self.model = init_model.model
            self.pool = init_model.pool

    def forward(self, x):
        # 提取卷积特征
        x = self.model.features(x)
        # 添加 SE 注意力模块
        x = self.se(x)
        # 根据设置的池化方式进行全局池化
        if self.pool == 'avg+max':
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            x = torch.cat((x1, x2), dim=1)
        elif self.pool == 'avg':
            x = self.model.avgpool2(x)
        elif self.pool == 'max':
            x = self.model.maxpool2(x)
        elif self.pool == 'gem':
            x = self.model.gem2(x)

        x = x.view(x.size(0), x.size(1))
        # L2 归一化（对检索任务有帮助）
        x = F.normalize(x, p=2, dim=1)
        return x


# Define the ResNet50-based Model
class ft_net(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net, self).__init__()
        vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        vit.head = nn.Identity()
        self.model = MultiScalePoolViT(vit)
        self.in_channels = vit.embed_dim * 2

        self.pool = pool
        if pool == 'gem':
            self.model.gem2 = GeM(dim=self.in_channels)

        self.se = nn.Identity()

        if init_model is not None:
            self.model = init_model.model
            self.pool = init_model.pool
            self.in_channels = init_model.in_channels

    def forward(self, x):
        feat = self.model(x)
        return feat


class two_view_net(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, pool='avg', share_weight=False):
        super().__init__()
        # two parallel ensembles (optionally weight‐shared)
        self.model_1 = MultiBackboneEnsemble(class_num, droprate, stride, pool)
        self.model_2 = self.model_1 if share_weight else MultiBackboneEnsemble(class_num, droprate, stride, pool)

        # metric head (ArcFace)
        # proj = [
        #   0: Linear(Dp → 0.75Dp),
        #   1: ReLU,
        #   2: Dropout,
        #   3: Linear(0.75Dp → total_dim//2),
        #   4: BatchNorm1d
        # ]
        embed_size = self.model_1.proj[3].out_features  # = total_dim//2
        self.metric_fc = ArcMarginProduct(
            in_features=embed_size,
            out_features=class_num,
            s=30.0,
            m=0.5
        )
        self.triplet_loss = TripletLoss(margin=0.3)

    def forward(self, x1, x2, labels1=None, labels2=None, x_a=None, x_p=None, x_n=None):
        # 1) compute embeddings
        z1 = self.model_1(x1) if x1 is not None else None   # [B, embed_size]
        z2 = self.model_2(x2) if x2 is not None else None   # [B, embed_size]

        # 2) ArcFace logits
        logits1 = self.metric_fc(z1, labels1) if z1 is not None else None
        logits2 = self.metric_fc(z2, labels2) if z2 is not None else None

        # 3) optional triplet loss
        if x_a is not None and x_p is not None and x_n is not None:
            a   = self.model_1(x_a)
            p   = self.model_1(x_p)
            n   = self.model_1(x_n)
            tri = self.triplet_loss(a, p, n)
            return logits1, logits2, tri

        # 4) keep original interface: only two ArcFace logits
        return logits1, logits2


class three_view_net(nn.Module):
    def __init__(self, class_num, droprate, stride=2, pool='avg',
                 share_weight=False, VGG16=False, circle=False):
        super(three_view_net, self).__init__()
        # backbone 1/2/3
        self.model_1 = ft_net(class_num, droprate, stride, pool=pool)
        self.model_2 = ft_net(class_num, droprate, stride, pool=pool)
        self.model_3 = self.model_1 if share_weight else ft_net(class_num, droprate, stride, pool=pool)
        self.circle  = circle

        # **同样动态计算 classifier 输入维度**
        in_feat = self.model_1.in_channels
        if pool == 'avg+max':
            in_feat *= 2
        self.classifier = ClassBlock(in_feat, class_num, droprate, return_f=circle)

    def forward(self, x1, x2, x3, x4=None):
        outs = []
        for x, model in zip([x1,x2,x3], [self.model_1,self.model_2,self.model_3]):
            if x is None:
                outs.append(None)
            else:
                f = model(x)
                outs.append(self.classifier(f))
        if x4 is None:
            return tuple(outs)
        # x4 走第二个 backbone
        f4 = self.model_2(x4)
        return (*tuple(outs), self.classifier(f4))

'''
# debug model structure
# Run this code with:
python model.py
'''
# === Improved __main__ test script ===
if __name__ == '__main__':
    net = two_view_net(751, droprate=0.5)
    # Dummy inputs and labels
    imgs = torch.randn(8, 3, 224, 224)
    labels = torch.zeros(8, dtype=torch.long)
    logits1, logits2 = net(imgs, imgs, labels, labels)
    print('Logits1 shape:', logits1.shape)
    print('Logits2 shape:', logits2.shape)
