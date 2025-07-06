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
class MultiScalePoolViT(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.vit = backbone
        self.gem = GeM(dim=self.vit.embed_dim)
        self.dropout = nn.Dropout(0.2)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # 1. Patch embedding
        x = self.vit.patch_embed(x)               # [B, N, C]
        B = x.size(0)
        # 2. Prepend class token
        cls_tokens = self.vit.cls_token.expand(B, -1, -1)  # [B,1,C]
        x = torch.cat((cls_tokens, x), dim=1)     # [B,1+N,C]
        # 3. Add positional embeddings and dropout
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        # 4. Transformer blocks
        for blk in self.vit.blocks:
            x = blk(x)

        # 5. Separate class token and patches
        class_tok = x[:, 0]                       # [B, C]
        patches = x[:, 1:].transpose(1, 2)
        N = patches.size(-1)
        H = W = int(math.sqrt(N))
        patches = patches.view(B, self.vit.embed_dim, H, W)

        # 6. GeM pooling on patch features
        ppool = self.gem(patches)                 # [B, C]

        # 7. Learnable fusion
        c = class_tok * self.alpha
        p = ppool * (1.0 - self.alpha)
        feat = torch.cat([c, p], dim=1)           # [B, 2C]
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
        vit.head = nn.Identity()                     # 去掉原分类头
        self.model = MultiScalePoolViT(vit)          # 多尺度聚合：[class_token, patch_pool] -> [2C]
        self.in_channels = vit.embed_dim * 2         # 2 × 768 = 1536

        # 如果仍想在聚合后加 GeM（可选）
        self.pool = pool
        if pool == 'gem':
            self.model.gem2 = GeM(dim=self.in_channels)

        # SEBlock 对一维特征不适用，直接跳过
        self.se = nn.Identity()

        # 若加载已有模型，继承其属性
        if init_model is not None:
            self.model       = init_model.model
            self.pool        = init_model.pool
            self.in_channels = init_model.in_channels

    def forward(self, x):
        # MultiScalePoolViT 已经输出 L2 归一化后的 [B, 2C] 特征
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
