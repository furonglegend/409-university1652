import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.models import convnext_base

######################################################################


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
        # —— ConvNeXt-Base 替换 ResNet50 —— 
        backbone = convnext_base(pretrained=True)
        backbone.classifier = nn.Identity()           # 去掉原分类头
        self.in_channels = 1024                        # ConvNeXt-Base 最后通道数
        self.pool = pool
        self.model = backbone
        self.se    = SEBlock(channel=self.in_channels)
        if pool == 'gem':
            self.model.gem2 = GeM(dim=self.in_channels)
        if init_model is not None:
            # 如果从已有模型继续 fine-tune，继承它的设置
            self.model       = init_model.model
            self.pool        = init_model.pool
            self.in_channels = init_model.in_channels

    def forward(self, x):
        x = self.model.features(x)
        x = self.se(x)
        if   self.pool == 'avg+max':
            a = F.adaptive_avg_pool2d(x, (1,1))
            b = F.adaptive_max_pool2d(x, (1,1))
            x = torch.cat([a,b], dim=1)
        elif self.pool == 'avg':
            x = F.adaptive_avg_pool2d(x, (1,1))
        elif self.pool == 'max':
            x = F.adaptive_max_pool2d(x, (1,1))
        elif self.pool == 'gem':
            x = self.model.gem2(x)
        x = x.view(x.size(0), -1)
        x = F.normalize(x, p=2, dim=1)
        return x


class two_view_net(nn.Module):
    def __init__(self, class_num, droprate, stride=2, pool='avg',
                 share_weight=False, VGG16=False, circle=False):
        super(two_view_net, self).__init__()
        # view1 / view2 backbone
        self.model_1 = ft_net(class_num, droprate, stride, pool=pool)
        self.model_2 = self.model_1 if share_weight else ft_net(class_num, droprate, stride, pool=pool)
        self.circle  = circle

        # **动态计算 classifier 输入维度**
        in_feat = self.model_1.in_channels
        if pool == 'avg+max':
            in_feat *= 2
        self.classifier = ClassBlock(in_feat, class_num, droprate, return_f=circle)

    def forward(self, x1, x2):
        y1 = y2 = None
        if x1 is not None:
            f1 = self.model_1(x1)
            y1 = self.classifier(f1)
        if x2 is not None:
            f2 = self.model_2(x2)
            y2 = self.classifier(f2)
        return y1, y2



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
if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it. 
    net = two_view_net(751, droprate=0.5, VGG16=True)
    #net.classifier = nn.Sequential()
    print(net)
    input = Variable(torch.FloatTensor(8, 3, 256, 256))
    output,output = net(input,input)
    print('net output size:')
    print(output.shape)