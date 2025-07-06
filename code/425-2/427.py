import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.models import convnext_base
import timm

######################################################################

# 在文件头部引入
import math
class RMAC(nn.Module):
    def __init__(self, levels=3):
        super().__init__()
        self.levels = levels

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        feats = []
        for l in range(1, self.levels+1):
            # 每层窗口大小
            wl = int(2 * min(H, W) / (l+1))
            hl = wl
            stride_h = math.floor((H - hl) / l) + 1
            stride_w = math.floor((W - wl) / l) + 1
            # unfold 生成所有窗口
            patches = x.unfold(2, hl, stride_h).unfold(3, wl, stride_w)
            # patches: [B,C,n1,n2,hl,wl] -> 合并成一维
            B,C,n1,n2,hl,wl = patches.shape
            patches = patches.contiguous().view(B, C, -1, hl, wl)
            # 对每个窗口做 GeM (或 avg)
            for i in range(patches.shape[2]):
                feats.append(F.adaptive_avg_pool2d(patches[:,:,i],1).view(B,C))
        # 拼所有区域特征并 normalize
        rmac_feat = torch.cat(feats, dim=1)  # [B, C * num_regions]
        return F.normalize(rmac_feat, dim=1)

class MultiScalePoolViT(nn.Module):
    def __init__(self, backbone, rmac_levels=3, dropout=0.2):
        super().__init__()
        self.vit = backbone
        # 保存通道数
        self.embed_dim = backbone.embed_dim  # C

        # 全局 GeM 池化
        self.gem  = GeM(dim=self.embed_dim)
        # 局部 RMAC 池化
        self.rmac = RMAC(levels=rmac_levels)

        # 计算 RMAC 输出维度：每层 l×l 个窗口
        num_regions = sum(l * l for l in range(1, rmac_levels + 1))
        local_dim    = self.embed_dim * num_regions

        # 把 local_feat 投影回 C 维
        self.local_reduce = nn.Linear(local_dim, self.embed_dim)

        # 三路特征拼接后做 MLP → 3 个融合权重 (softmax)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(3 * self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, 3),
            nn.Softmax(dim=1)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B = x.size(0)
        # 1）ViT 前向到最后一个 Transformer block
        x = self.vit.patch_embed(x)                                # [B, N, C]
        cls_tok = self.vit.cls_token.expand(B, -1, -1)             # [B,1,C]
        x = torch.cat([cls_tok, x], dim=1)                         # [B,1+N,C]
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        for blk in self.vit.blocks:
            x = blk(x)

        # 2）解出 class token 和 patch 特征图
        class_tok = x[:, 0]                                        # [B, C]
        patches   = x[:, 1:].transpose(1, 2)                       # [B, C, N]
        C = patches.size(1)                                        # 动态获取通道数
        N = patches.size(-1)
        H = W = int(math.sqrt(N))
        feat_map = patches.view(B, C, H, W)                        # [B, C, H, W]

        # 3）全局 GeM
        global_feat = self.gem(feat_map)                           # [B, C]
        # 4）局部 RMAC
        raw_local  = self.rmac(feat_map)                           # [B, C * num_regions]
        local_feat = self.local_reduce(raw_local)                  # [B, C]

        # 5）三路融合
        concat_feats = torch.cat([class_tok, global_feat, local_feat], dim=1)  # [B, 3C]
        weights      = self.fusion_mlp(concat_feats)                         # [B, 3]
        w_cls, w_g, w_l = weights[:,0:1], weights[:,1:2], weights[:,2:3]     # each [B,1]

        fused = w_cls * class_tok + w_g * global_feat + w_l * local_feat     # [B, C]
        fused = self.dropout(fused)

        # 6）L2 归一化
        return F.normalize(fused, dim=1)                                     # [B, C]
        
class ArcMarginProduct(nn.Module):
    """ArcFace with learnable (per-class) margin."""
    def __init__(self, in_features, out_features, s=64.0, m=0.50, per_class_m=True):
        super().__init__()
        # weight for classification
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        # scale (固定)  
        self.s = s
        # margin: 如果 per_class_m=True，则对每个类学习一个 margin，否则学习一个全局 margin
        if per_class_m:
            self.m = nn.Parameter(torch.ones(out_features) * m, requires_grad=True)  # [out_features]
        else:
            self.m = nn.Parameter(torch.tensor(m), requires_grad=True)                # scalar

    def forward(self, x, labels):
        # x: [B, C], labels: [B]
        # 1) normalize features & weights
        x_norm   = F.normalize(x)                              # [B, C]
        w_norm   = F.normalize(self.weight)                    # [out_features, C]
        cosine   = F.linear(x_norm, w_norm)                    # [B, out_features]
        sine     = torch.sqrt(1.0 - torch.clamp(cosine**2, 0, 1))

        # 2) compute phi = cos(θ + m) = cosθ·cos m - sinθ·sin m
        #    当 m 是向量时，cos(m) 和 sin(m) 会 broadcast 到 [B, out_features]
        cos_m = torch.cos(self.m)
        sin_m = torch.sin(self.m)
        phi = cosine * cos_m.unsqueeze(0) - sine * sin_m.unsqueeze(0)

        # 3) 只在 ground-truth 类上用 phi，其他类保持原 cosine
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1,1), 1.0)

        # 4) scale 输出
        logits = self.s * (one_hot * phi + (1.0 - one_hot) * cosine)
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
    # GeM zhedong zheng
    def __init__(self, dim = 2048, p=3, eps=1e-6):
        super(GeM,  self).__init__()
        self.p = nn.Parameter(torch.ones(dim)*p, requires_grad = True) #initial p
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
        x = x.pow(1./p)
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
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
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
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x,f
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
        # —— Vision Transformer + Multi-Scale Pooling Backbone —— 
        vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        vit.head = nn.Identity()                      # 去掉原分类头
        self.model = MultiScalePoolViT(vit)           # 多尺度聚合，现在输出 [B, C]
        
        # 修改：in_channels 应与 MultiScalePoolViT 输出维度相同
        self.in_channels = vit.embed_dim              # 原来是 vit.embed_dim * 2

        # 如果仍想在聚合后加 GeM（可选）
        self.pool = pool
        if pool == 'gem':
            # gem2 也要用新的 in_channels
            self.model.gem2 = GeM(dim=self.in_channels)

        # SEBlock 对一维特征不适用，直接跳过
        self.se = nn.Identity()

        # 若加载已有模型，继承其属性
        if init_model is not None:
            self.model       = init_model.model
            self.pool        = init_model.pool
            self.in_channels = init_model.in_channels

    def forward(self, x):
        # MultiScalePoolViT 已经输出 L2 归一化后的 [B, C] 特征
        feat = self.model(x)
        return feat


class two_view_net(nn.Module):
    def __init__(self, class_num, droprate, stride=2, pool='avg', share_weight=False):
        super().__init__()
        self.model_1 = ft_net(class_num, droprate, stride, pool=pool)
        self.model_2 = self.model_1 if share_weight else ft_net(class_num, droprate, stride, pool=pool)
        self.proj_head = nn.Sequential(
            nn.Linear(self.model_1.in_channels, self.model_1.in_channels),
            nn.BatchNorm1d(self.model_1.in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=droprate),
        )
        self.metric_fc = ArcMarginProduct(self.model_1.in_channels, class_num)

    def forward(self, x1, x2, labels1=None, labels2=None):
        logits1 = logits2 = None

        if x1 is not None:
            f1 = self.model_1(x1)                # [B, 2C]
            z1 = self.proj_head(f1)
            logits1 = self.metric_fc(z1, labels1)

        if x2 is not None:
            f2 = self.model_2(x2)                # [B, 2C]
            z2 = self.proj_head(f2)             # apply same projection
            logits2 = self.metric_fc(z2, labels2)

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
