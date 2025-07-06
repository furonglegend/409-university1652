import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as Fimport torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F
import torch.nn.functional as F
######################################################################


# 添加DenseNet实现
class ft_net_DenseNet(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, pool='avg'):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.pool = pool
        if pool == 'gem':
            self.gem = GeM(dim=1024)
        self.model = model_ft

    def forward(self, x):
        x = self.model.features(x)
        if self.pool == 'gem':
            x = self.gem(x)
        x = x.view(x.size(0), -1)  # 确保展平操作
        print(f"DenseNet特征维度: {x.shape}")  # 调试输出
        return x
# GeM池化层


# 在backbone中添加SE注意力模块
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
        # 添加SE模块
        self.se = SELayer(planes * self.expansion, reduction)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        # 在最后一个BN之后、残差连接之前添加SE
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class GeM(nn.Module):
    def __init__(self, dim=2048, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(dim) * p, requires_grad=True)  # 初始p值
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
# 使用kaiming normal初始化权重
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
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
# 分类块，包含批量归一化、Dropout等
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, 
                num_bottleneck=1024, linear=True, 
                return_f=False, bnorm=True, relu=True):
        super().__init__()
        self.return_f = return_f
        layers = []
        current_dim = input_dim

        # 第一阶段：线性投影
        if linear:
            layers += [
                nn.Linear(current_dim, num_bottleneck),
                nn.BatchNorm1d(num_bottleneck) if bnorm else nn.Identity(),
                nn.LeakyReLU(0.1, inplace=True) if relu else nn.Identity(),
                nn.Dropout(p=0.5)
            ]
            current_dim = num_bottleneck  # 更新当前维度

        # 第二阶段：特征细化
        refinement = [
            nn.Linear(current_dim, current_dim//2),
            nn.BatchNorm1d(current_dim//2) if bnorm else nn.Identity(),
            nn.LeakyReLU(0.1, inplace=True) if relu else nn.Identity(),
            nn.Dropout(p=droprate)
        ]
        layers += refinement
        current_dim = current_dim // 2

        self.add_block = nn.Sequential(*layers)
        self.classifier = nn.Linear(current_dim, class_num)

        # 初始化
        self.add_block.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        features = self.add_block(x)
        if self.return_f:
            return self.classifier(features), features
        return self.classifier(features)
class ft_net_VGG16(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net_VGG16, self).__init__()
        model_ft = models.vgg16_bn(pretrained=True)

        self.pool = pool
        if pool == 'avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))
        elif pool == 'avg':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        elif pool == 'max':
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))
        elif pool == 'gem':
            model_ft.gem2 = GeM(dim=512)

        self.model = model_ft

        if init_model is not None:
            self.model = init_model.model
            self.pool = init_model.pool

    def forward(self, x):
        x = self.model.features(x)
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
        print(f"VGG16特征维度: {x.shape}")  # 调试输出
        return x

# Define the ResNet50-based Model
class ft_net(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1, 1)
            model_ft.layer4[0].conv2.stride = (1, 1)

        self.pool = pool
        if pool == 'avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))
        elif pool == 'avg':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        elif pool == 'max':
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))
        elif pool == 'gem':
            model_ft.gem2 = GeM(dim=2048)

        self.model = model_ft

        if init_model is not None:
            self.model = init_model.model
            self.pool = init_model.pool

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
            x = torch.cat((x1, x2), dim=1)
        elif self.pool == 'avg':
            x = self.model.avgpool2(x)
        elif self.pool == 'max':
            x = self.model.maxpool2(x)
        elif self.pool == 'gem':
            x = self.model.gem2(x)

        x = x.view(x.size(0), x.size(1))
        print(f"ResNet特征维度: {x.shape}")  # 调试输出
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


class three_view_net(nn.Module):
    def __init__(self, class_num, droprate, stride=2, pool='avg', share_weight=False, 
                 VGG16=False, ResNet50=False, DenseNet=False, Inception=False, circle=False):
        super().__init__()
        self.target_dim = 2048
        self.models_list = nn.ModuleList()
        
        # 初始化各Backbone并统一特征维度
        def add_model(model_class, in_dim):
            model = model_class(class_num, droprate, stride, pool=pool)
            if in_dim != self.target_dim:  # 需要投影
                return nn.Sequential(model, Projection(in_dim, self.target_dim))
            return model
        
        if VGG16:  # VGG输出512维
            self.models_list.append(add_model(ft_net_VGG16, 512))
        if ResNet50:  # ResNet输出2048维
            self.models_list.append(add_model(ft_net, 2048))
        if DenseNet:  # DenseNet输出1024维
            self.models_list.append(add_model(ft_net_DenseNet, 1024))
        if Inception:  # Inception输出2048维
            self.models_list.append(add_model(ft_net_Inception, 2048))
            
        # 多模型融合层
        self.model_ensemble = ModelAveraging(self.models_list)
        
        # 动态权重融合模块
        self.fusion = nn.Sequential(
            nn.Linear(self.target_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 3),
            nn.Softmax(dim=1)
        )
        
        # 分类器（统一输入维度为2048）
        self.classifier = ClassBlock(
            input_dim=self.target_dim,
            class_num=class_num,
            droprate=droprate,
            return_f=circle
        )

    def forward(self, x1, x2, x3, x4=None):
        # 特征提取（确保输出为张量）
        f1 = self.model_ensemble(x1)
        f2 = self.model_ensemble(x2)
        f3 = self.model_ensemble(x3)
        
        # 动态加权融合
        combined = torch.stack([f1, f2, f3], dim=1)  # [bs,3,2048]
        weights = self.fusion(combined.mean(dim=2))   # [bs,3]
        fused_feature = (combined * weights.unsqueeze(2)).sum(dim=1)  # [bs,2048]
        
        # 分类输出
        return self.classifier(fused_feature)
# 多视图网络，用于处理多个图像模态的比较
# model.py 关键修正
class multi_view_net(nn.Module):
    def __init__(self, class_num, droprate, stride=2, pool='avg', 
                share_weight=False, resnet=False, vgg=False, densenet=False):
        super().__init__()
        self.backbones = nn.ModuleList()
        self.feat_dims = []

        # 准确计算各Backbone特征维度
        def get_feat_dim(model_type):
            dim_map = {'resnet':2048, 'vgg':512, 'densenet':1024}
            base_dim = dim_map[model_type]
            return base_dim * (2 if 'avg+max' in pool else 1)  # 单视图维度

        if resnet:
            self.backbones.append(ft_net(class_num, droprate, stride, pool))
            self.feat_dims.append(get_feat_dim('resnet')*2)  # 双视图拼接
        if vgg:
            self.backbones.append(ft_net_VGG16(...))
            self.feat_dims.append(get_feat_dim('vgg')*2)
        if densenet:
            self.backbones.append(ft_net_DenseNet(...))
            self.feat_dims.append(get_feat_dim('densenet')*2)

        # 分类器维度适配
        total_feat_dim = sum(self.feat_dims)
        self.fusion = nn.Linear(total_feat_dim, 2048) if len(self.backbones)>1 else None
        self.classifier = ClassBlock(2048 if self.fusion else total_feat_dim, class_num, droprate)

    def forward(self, x1, x2):  # 固定双视图输入
        features = []
        for model in self.backbones:
            feat1 = model(x1)
            feat2 = model(x2)
            combined = torch.cat([feat1, feat2], dim=1)
            features.append(combined)
        
        fused = torch.cat(features, dim=1) if len(features)>1 else features[0]
        if self.fusion:
            fused = self.fusion(fused)
        return self.classifier(fused)
'''
# debug model structure
# Run this code with:
python model.py
'''
# 测试新的多视图模型
# 测试代码应验证各维度
# Enhanced test cases
if __name__ == '__main__':
    test_configs = [
        # ResNet单backbone
        {'resnet':True, 'vgg':False, 'densenet':False, 'pool':'gem', 'expected_dim':751},
        {'resnet':True, 'vgg':False, 'densenet':False, 'pool':'avg+max', 'expected_dim':751},
        # VGG单backbone
        {'resnet':False, 'vgg':True, 'densenet':False, 'pool':'gem', 'expected_dim':751},
        # 多backbone组合
        {'resnet':True, 'vgg':True, 'densenet':False, 'pool':'gem', 'expected_dim':751}
    ]

    for config in test_configs:
        model = multi_view_net(
            class_num=751,
            droprate=0.5,
            stride=2,
            pool=config['pool'],
            resnet=config['resnet'],
            vgg=config['vgg'],
            densenet=config['densenet']
        )
        
        # 前向测试
        input_tensor = torch.randn(8, 3, 256, 256)
        output = model(input_tensor, input_tensor)[0]
        assert output.shape == (8, 751), f"维度错误: 预期(8,751), 实际{output.shape}"
        print(f"测试通过: {config}")