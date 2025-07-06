import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F


######################################################################


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=8):
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


# model.py中GeM类的修改
class GeM(nn.Module):
    def __init__(self, dim=1024, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.dim = dim
        # 初始化可学习参数为标量（而非向量）
        self.p = nn.Parameter(torch.tensor(p))  
        self.eps = eps

    def forward(self, x):
        # 确保p为标量，适用于所有通道
        p = self.p.expand(1, self.dim, 1, 1)  # 形状变为 [1, dim, 1, 1]
        # 逐元素幂运算
        x_powed = x.clamp(min=self.eps).pow(p)
        # 全局平均池化
        pooled = F.adaptive_avg_pool2d(x_powed, (1, 1))
        # 反幂运算
        return pooled.pow(1./p)

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
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=1024, linear=True, return_f=False,temperature=0.05):
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
        add_block.apply(weights_init_kaiming)  # 显式调用初始化

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)  # 显式调用初始化

        self.add_block = add_block
        self.classifier = classifier
        self.temperature = temperature
    def forward(self, x):
        x = self.add_block(x)
        return self.classifier(x / self.temperature)


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
            model_ft.gem2 = GeM(dim=1024)

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
        #x = F.normalize(x, p=2, dim=1)
        return x


# Define the ResNet50-based Model
class ft_net(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net, self).__init__()
        
        # 加载ConvNeXt预训练模型
        model_ft = models.convnext_base(pretrained=True)
        
        # 特征提取器重构（保留空间维度）
        self.features = nn.Sequential(
            *list(model_ft.children())[:-2]  # 移除最后两个模块（分类头和全局池化）
        )
        
        # 池化配置
        self.pool = pool
        self._configure_pooling(pool)
        
        # 注意力模块（输入通道数需匹配ConvNeXt输出）
        self.se = SEBlock(channel=1024)  # ConvNeXt-base最终特征维度1024
        
        # 自适应全局池化（保留空间维度）
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 输出[B, 1024, 1, 1]
        
        # 特征增强层
        self.fc = self._build_fc(droprate)
        
        # 初始化参数
        if init_model:
            self.load_state_dict(init_model.state_dict())

    def _configure_pooling(self, pool_type):
        """动态配置池化策略"""
        if pool_type == 'avg+max':
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.maxpool = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'gem':
            self.gem = GeM(dim=1024)
        else:
            self.pool_type = pool_type  # 记录池化类型用于前向传播

    def _build_fc(self, droprate):
        """构建特征增强全连接层"""
        return nn.Sequential(
            nn.Linear(1024, 1024, bias=False),
            nn.GELU(),
            nn.Dropout(droprate),
            nn.LayerNorm(1024)
        )

    def forward(self, x):
        # 特征提取（保持四维）
        x = self.features(x)  # 输出形状: [B, 1024, H, W]
        
        # 应用SE注意力（输入四维）
        x = self.se(x)
        
        # 池化处理
        if self.pool == 'avg+max':
            x_avg = self.avgpool(x).flatten(1)  # [B, 1024]
            x_max = self.maxpool(x).flatten(1)  # [B, 1024]
            x = torch.cat([x_avg, x_max], dim=1)  # [B, 2048]
        elif self.pool == 'gem':
            x = self.gem(x).flatten(1)  # [B, 1024]
        else:  # avg或max
            x = self.global_pool(x).flatten(1)  # [B, 1024]
        
        # 特征增强
        x = self.fc(x)
        
        # L2归一化
        return F.normalize(x, p=2, dim=1)

class two_view_net(nn.Module):
    def __init__(self, class_num, droprate, stride=2, pool='avg', share_weight=False, circle=False):
        super(two_view_net, self).__init__()
        # 使用 ConvNeXt 作为骨干网络
        self.model_1 = ft_net(class_num, stride=stride, pool=pool)
        if share_weight:
            self.model_2 = self.model_1
        else:
            self.model_2 = ft_net(class_num, stride=stride, pool=pool)

        self.circle = circle
        # 增加特征融合模块
        self.fusion = nn.Sequential(
            nn.Linear(2048, 1024),  # 适应 ConvNeXt 输出特征维度
            nn.ReLU(),
            nn.Dropout(p=droprate)
        )

        self.classifier = ClassBlock(1024, class_num, droprate, return_f=circle)

    def forward(self, x1, x2):
        if x1 is None:
            y1 = None
        else:
            x1 = self.model_1(x1)
            x1 = x1.view(x1.size(0), -1)

        if x2 is None:
            y2 = None
        else:
            x2 = self.model_2(x2)
            x2 = x2.view(x2.size(0), -1)

        if x1 is not None and x2 is not None:
            x = torch.cat((x1, x2), dim=1)
            x = self.fusion(x)
            y = self.classifier(x)
            return y, y
        else:
            if x1 is not None:
                y = self.classifier(x1)
                return y, y
            else:
                y = self.classifier(x2)
                return y, y



class three_view_net(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, pool='avg', share_weight=False, circle=False):  # 移除了VGG16参数
        super(three_view_net, self).__init__()
        
        # ConvNeXt配置参数
        self.base_dim = 1024  # ConvNeXt-base输出维度
        self.pool = pool
        
        # 初始化ConvNeXt骨干网络
        model_cls = ft_net  # 使用ConvNeXt的实现

        # 初始化三个子模型 --------------------------------------------------
        self.model_1 = model_cls(
            class_num=class_num,
            droprate=droprate,
            stride=stride,
            pool=pool
        )
        
        # 权重共享配置
        if share_weight:
            self.model_2 = self.model_1
            self.model_3 = self.model_1
        else:
            self.model_2 = model_cls(
                class_num=class_num,
                droprate=droprate,
                stride=stride,
                pool=pool
            )
            self.model_3 = model_cls(
                class_num=class_num,
                droprate=droprate,
                stride=stride,
                pool=pool
            )

        # 动态维度计算 ------------------------------------------------------
        if self.pool == 'avg+max':
            input_dim = self.base_dim * 2  # 双池化拼接维度翻倍
        else:
            input_dim = self.base_dim

        # 统一分类器初始化
        self.classifier = ClassBlock(
            input_dim=input_dim,
            class_num=class_num,
            droprate=droprate,
            num_bottleneck=input_dim,  # 根据输入维度动态调整 [关键修复]
            return_f=circle
        )

    def forward(self, x1, x2, x3, x4=None):
        """ 前向传播支持三种视图输入 + 可选的第四视图 """
        
        def process_input(model, x):
            """ 统一处理输入: 提取特征 -> 分类 """
            if x is None:
                return None
            features = model(x)           # 特征提取
            return self.classifier(features)

        # 处理三个主视图
        y1 = process_input(self.model_1, x1)  # 卫星视图
        y2 = process_input(self.model_2, x2)  # 街景视图
        y3 = process_input(self.model_3, x3)  # 无人机视图

        # 可选第四视图处理
        if x4 is not None:
            y4 = process_input(self.model_2, x4)  # 共享街景模型处理
            return y1, y2, y3, y4
        
        return y1, y2, y3

'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
    # 使用 ConvNeXt 作为骨干网络，VGG16 应该被替换为 False
    net = two_view_net(751, droprate=0.5, stride=2, pool='avg', share_weight=False, circle=False)
    
    # 打印模型架构
    print(net)
    
    # 测试模型的前向传播
    input = Variable(torch.FloatTensor(8, 3, 256, 256))  # 假设 batch_size = 8, 输入为 256x256 RGB 图像
    output, output = net(input, input)  # 两个视角的输入
    print('net output size:')
    print(output.shape)  # 打印输出的形状
