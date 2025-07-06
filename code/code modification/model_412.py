import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F
from math import sqrt

######################################################################
# 新增扩散模型组件（修正残差连接）
class DiffusionStep(nn.Module):
    """单步扩散特征增强模块（含残差连接）"""
    def __init__(self, feat_dim=2048, time_dim=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.feature_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim//2),
            nn.GroupNorm(8, feat_dim//2),
            nn.SiLU()
        )
        self.noise_pred = nn.Sequential(
            nn.Linear(feat_dim//2 + time_dim, feat_dim),
            nn.GroupNorm(8, feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, x, t):
        residual = x  # 残差连接
        t_embed = self.time_embed(t.view(-1,1))
        h = self.feature_proj(x)
        h = torch.cat([h, t_embed], dim=1)
        pred = self.noise_pred(h)
        return residual + pred  # 添加残差

class DiffusionEnhancer(nn.Module):
    """多步扩散特征增强器（修正实现）"""
    def __init__(self, feat_dim=2048, num_steps=4):
        super().__init__()
        self.steps = nn.ModuleList([
            DiffusionStep(feat_dim) for _ in range(num_steps)
        ])
        
    def forward(self, x):
        for step in self.steps:
            # 生成随机时间步参数
            t = torch.rand(x.size(0), device=x.device)  # [0,1)随机时间
            x = step(x, t)
        return x

######################################################################
# 特征对齐模块（修正维度处理）
class FeatureAligner(nn.Module):
    """跨视角特征对齐（输出三个独立特征）"""
    def __init__(self, feat_dim):
        super().__init__()
        # 三个独立的注意力机制
        self.attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim*3, feat_dim),
                nn.ReLU(),
                nn.Linear(feat_dim, 1),
                nn.Sigmoid()
            ) for _ in range(3)
        ])
        
    def forward(self, feat1, feat2, feat3):
        # 每个特征的注意力权重计算
        attn1 = self.attention[0](torch.cat([feat1, feat2, feat3], dim=1))
        attn2 = self.attention[1](torch.cat([feat1, feat2, feat3], dim=1))
        attn3 = self.attention[2](torch.cat([feat1, feat2, feat3], dim=1))
        
        # 加权融合
        aligned1 = attn1 * feat1 + feat2 + feat3
        aligned2 = feat1 + attn2 * feat2 + feat3
        aligned3 = feat1 + feat2 + attn3 * feat3
        
        return aligned1, aligned2, aligned3


class ft_net_VGG16(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net_VGG16, self).__init__()
        model_ft = models.vgg16_bn(pretrained=True)
        # avg pooling to global pooling
        #if stride == 1:
        #    model_ft.layer4[0].downsample[0].stride = (1,1)
        #    model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        if pool =='avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            #self.classifier = ClassBlock(4096, class_num, droprate)
        elif pool=='avg':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            #self.classifier = ClassBlock(2048, class_num, droprate)
        elif pool=='max':
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
        elif pool=='gem':
            model_ft.gem2 = GeM(dim = 512)

        self.model = model_ft

        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        x = self.model.features(x)
        if self.pool == 'avg+max':
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            x = torch.cat((x1,x2), dim = 1)
        elif self.pool == 'avg':
            x = self.model.avgpool2(x)
        elif self.pool == 'max':
            x = self.model.maxpool2(x)
        elif self.pool=='gem':
            x = self.model.gem2(x)

        x = x.view(x.size(0), x.size(1))

        #x = self.classifier(x)
        return x
 # Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        if pool =='avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            #self.classifier = ClassBlock(4096, class_num, droprate)
        elif pool=='avg':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            #self.classifier = ClassBlock(2048, class_num, droprate)
        elif pool=='max':
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
        elif pool=='gem':
            model_ft.gem2 = GeM(dim=2048)

        self.model = model_ft

        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        if self.pool == 'avg+max':
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            x = torch.cat((x1,x2), dim = 1)
        elif self.pool == 'avg':
            x = self.model.avgpool2(x)
        elif self.pool == 'max':
            x = self.model.maxpool2(x)
        elif self.pool == 'gem':
            x = self.model.gem2(x)

        x = x.view(x.size(0), x.size(1))
        #x = self.classifier(x)
        return x
        
######################################################################
# 修改后的three_view_net（修正分类器结构）
class three_view_net(nn.Module):
    def __init__(self, class_num, droprate, stride=2, pool='avg', 
                share_weight=False, VGG16=False, circle=False,
                use_diffusion=True):
        super().__init__()
        self.VGG16 = VGG16
        self.feat_dim = 512 if VGG16 else 2048  # 动态特征维度
        
        # 基础特征提取网络
        base_model = ft_net_VGG16 if VGG16 else ft_net
        self.model_1 = base_model(class_num, droprate, stride, pool=pool)
        self.model_2 = base_model(class_num, droprate, stride, pool=pool)
        self.model_3 = base_model(class_num, droprate, stride, pool=pool)

        # 特征处理模块
        self.aligner = FeatureAligner(self.feat_dim)
        self.diffusion = DiffusionEnhancer(self.feat_dim) if use_diffusion else None
        
        # 分类器（保持各视角独立）
        self.classifier1 = ClassBlock(self.feat_dim, class_num, droprate)
        self.classifier2 = ClassBlock(self.feat_dim, class_num, droprate)
        self.classifier3 = ClassBlock(self.feat_dim, class_num, droprate)

    def forward(self, x1, x2, x3):
        # 特征提取
        f1 = self.model_1(x1)
        f2 = self.model_2(x2)
        f3 = self.model_3(x3)
        
        # 维度验证
        assert f1.size(1) == self.feat_dim, f"特征维度错误: 预期{self.feat_dim}, 实际{f1.size(1)}"
        
        # 特征对齐（直接返回三个特征）
        f1, f2, f3 = self.aligner(f1, f2, f3)  # 关键修改点
        
        # 扩散增强
        if self.diffusion:
            f1 = self.diffusion(f1)
            f2 = self.diffusion(f2)
            f3 = self.diffusion(f3)
        
        # 分类输出
        return (
            self.classifier1(f1),
            self.classifier2(f2),
            self.classifier3(f3)
        )
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
class two_view_net(nn.Module):
    def __init__(self, class_num, droprate, stride = 2, pool = 'avg', share_weight = False, VGG16=False, circle=False,):
        super(two_view_net, self).__init__()
        if VGG16:
            self.model_1 =  ft_net_VGG16(class_num, stride=stride, pool = pool)
        else:
            self.model_1 =  ft_net(class_num, stride=stride, pool = pool)
        if share_weight:
            self.model_2 = self.model_1
        else:
            if VGG16:
                self.model_2 =  ft_net_VGG16(class_num, stride = stride, pool = pool)
            else:
                self.model_2 =  ft_net(class_num, stride = stride, pool = pool)

        self.circle = circle

        self.classifier = ClassBlock(2048, class_num, droprate, return_f = circle)
        if pool =='avg+max':
            self.classifier = ClassBlock(4096, class_num, droprate, return_f = circle)
        if VGG16:
            self.classifier = ClassBlock(512, class_num, droprate, return_f = circle)
            if pool =='avg+max':
                self.classifier = ClassBlock(1024, class_num, droprate, return_f = circle)

    def forward(self, x1, x2):
        if x1 is None:
            y1 = None
        else:
            x1 = self.model_1(x1)
            y1 = self.classifier(x1)

        if x2 is None:
            y2 = None
        else:
            x2 = self.model_2(x2)
            y2 = self.classifier(x2)
        return y1, y2


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

# ... [保留其他原有结构 ft_net_VGG16, ft_net, ClassBlock 等] ...

if __name__ == '__main__':
    # 测试ResNet版本
    net_resnet = three_view_net(751, droprate=0.5, VGG16=False)
    x = torch.randn(2,3,256,256)
    outputs = net_resnet(x, x, x)
    print(f"ResNet输出维度: {[o.shape for o in outputs]}")  # 应得3个(2,751)

    # 测试VGG16版本
    net_vgg = three_view_net(751, droprate=0.5, VGG16=True)
    outputs_vgg = net_vgg(x, x, x)
    print(f"VGG16输出维度: {[o.shape for o in outputs_vgg]}")  # 应得3个(2,751)