import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F

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

        # 使用 RegNet 替代 ResNet50
        model_ft = models.regnet_y_16gf(pretrained=True)  # 使用 RegNet 作为骨干网络

        self.pool = pool
        if pool == 'avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))
        elif pool == 'avg':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        elif pool == 'max':
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))
        elif pool == 'gem':
            model_ft.gem2 = GeM(dim=2048)  # 修改为 RegNet 的输出维度

        self.model = model_ft
        # 添加 SE 模块，输入通道数为 2048
        self.se = SEBlock(channel=2048)  # 修改为 2048，以适应 RegNet 输出

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
        # 添加 SE 模块
        x = self.se(x)
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
        x = F.normalize(x, p=2, dim=1)
        return x

class two_view_net(nn.Module):
    def __init__(self, class_num, droprate, stride=2, pool='avg', share_weight=False, circle=False):
        super(two_view_net, self).__init__()

        # 使用 RegNet 作为骨干网络
        self.model_1 = ft_net(class_num, stride=stride, pool=pool)
        if share_weight:
            self.model_2 = self.model_1
        else:
            self.model_2 = ft_net(class_num, stride=stride, pool=pool)

        self.circle = circle

        # 修改 ClassBlock 输入维度为 2048
        self.classifier = ClassBlock(2048, class_num, droprate, return_f=circle)
        if pool == 'avg+max':
            self.classifier = ClassBlock(4096, class_num, droprate, return_f=circle)

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
    def __init__(self, class_num, droprate, stride = 2, pool = 'avg', share_weight = False, VGG16=False, circle=False):
        super(three_view_net, self).__init__()
        if VGG16:
            self.model_1 =  ft_net_VGG16(class_num, stride = stride, pool = pool)
            self.model_2 =  ft_net_VGG16(class_num, stride = stride, pool = pool)
        else:
            self.model_1 =  ft_net(class_num, stride = stride, pool = pool)
            self.model_2 =  ft_net(class_num, stride = stride, pool = pool)

        if share_weight:
            self.model_3 = self.model_1
        else:
            if VGG16:
                self.model_3 =  ft_net_VGG16(class_num, stride = stride, pool = pool)
            else:
                self.model_3 =  ft_net(class_num, stride = stride, pool = pool)

        self.circle = circle

        self.classifier = ClassBlock(2048, class_num, droprate, return_f = circle)
        if pool =='avg+max':

            self.classifier = ClassBlock(4096, class_num, droprate, return_f = circle)

    def forward(self, x1, x2, x3, x4 = None): # x4 is extra data
        if x1 is None:
            y1 = None
        else:
            x1 = self.model_1(x1)
            x1 = x1.view(x1.size(0), x1.size(1))
            y1 = self.classifier(x1)

        if x2 is None:
            y2 = None
        else:
            x2 = self.model_2(x2)
            x2 = x2.view(x2.size(0), x2.size(1))
            y2 = self.classifier(x2)

        if x3 is None:
            y3 = None
        else:
            x3 = self.model_3(x3)
            x3 = x3.view(x3.size(0), x3.size(1))
            y3 = self.classifier(x3)

        if x4 is None:
            return y1, y2, y3
        else:
            x4 = self.model_2(x4)
            x4 = x4.view(x4.size(0), x4.size(1))
            y4 = self.classifier(x4)
            return y1, y2, y3, y4


'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it. 
    net = two_view_net(751, droprate=0.5, VGG16=False)
    #net.classifier = nn.Sequential()
    print(net)
    input = Variable(torch.FloatTensor(8, 3, 256, 256))
    output,output = net(input,input)
    print('net output size:')
    print(output.shape)