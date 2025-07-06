import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F

######################################################################
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out,_ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))

class CBAM(nn.Module):
    def __init__(self, in_planes, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out


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
    def __init__(self, input_dim, class_num, droprate=0.5, return_f=True,
                 relu=False, bnorm=True, num_bottleneck=512, linear=True):
        super().__init__()
        self.return_f = return_f

        layers = []
        if linear:
            layers.append(nn.Linear(input_dim, num_bottleneck))
        if bnorm:
            layers.append(nn.BatchNorm1d(num_bottleneck))
        if relu:
            layers.append(nn.LeakyReLU(0.1))
        if droprate > 0:
            layers.append(nn.Dropout(droprate))
        self.add_block = nn.Sequential(*layers)
        self.add_block.apply(self._init_kaiming)

        self.classifier = nn.Linear(num_bottleneck, class_num)
        init.normal_(self.classifier.weight, std=0.001)
        init.constant_(self.classifier.bias, 0.0)

    def _init_kaiming(self, m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm1d):
            init.normal_(m.weight, 1.0, 0.02)
            init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            feat = x
            out = self.classifier(x)
            return out, feat
        else:
            return self.classifier(x)
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
        model_ft = models.resnet50(pretrained=True)
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        if pool == 'avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
        elif pool == 'avg':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
        elif pool == 'max':
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
        elif pool == 'gem':
            model_ft.gem2 = GeM(dim=2048)

        self.model = model_ft
        # 替换为 CBAM 模块，输入通道数为 2048
        self.cbam = CBAM(in_planes=2048)

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
        # 使用 CBAM 替换原来的 SE 模块
        x = self.cbam(x)
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

class ft_net_ResNeXt(nn.Module):
    def __init__(self, class_num, droprate=0.5, pool='avg', init_model=None):
        super(ft_net_ResNeXt, self).__init__()
        # 使用 torchvision 内置的 ResNeXt50_32x4d
        model_ft = models.resnext50_32x4d(pretrained=True)
        
        self.pool = pool
        if pool == 'avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
        elif pool == 'avg':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
        elif pool == 'max':
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
        elif pool == 'gem':
            model_ft.gem2 = GeM(dim=2048)

        self.model = model_ft
        # 将原来加入的 SE 模块替换为 CBAM 模块，输入通道数为 2048
        self.cbam = CBAM(in_planes=2048)
        
        if init_model is not None:
            self.model = init_model.model
            self.pool = init_model.pool

    def forward(self, x):
        # 基于 ResNeXt 的特征提取过程与 ResNet50 类似
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        # 使用 CBAM 注意力模块替换 SE 模块
        x = self.cbam(x)
        # 池化操作
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
        # L2 归一化
        x = F.normalize(x, p=2, dim=1)
        return x



class two_view_net(nn.Module):
    def __init__(self,
                 class_num: int,
                 droprate: float = 0.5,
                 pool: str = 'avg',
                 share_weight: bool = False,
                 circle: bool = False,
                 use_swin: bool = False,
                 use_resnext: bool = False):
        super().__init__()
        self.pool   = pool
        self.circle = circle

        # 1) 选择 Backbone1
        if use_swin:
            bb1 = models.swin_t(pretrained=True)
            bb1.head = nn.Identity()
            self.model_1 = bb1
            feat_dim = 768

        elif use_resnext:
            resx1 = models.resnext50_32x4d(pretrained=True)
            # 去掉原 classifier
            modules = list(resx1.children())[:-2]  # 保留到 layer4 输出
            self.model_1 = nn.Sequential(*modules)
            feat_dim = 2048

        else:
            res1 = models.resnet101(pretrained=True)
            self.model_1 = nn.Sequential(
                res1.conv1, res1.bn1, res1.relu, res1.maxpool,
                res1.layer1, res1.layer2, res1.layer3, res1.layer4
            )
            feat_dim = 2048

        # 2) Backbone2 （共享或重新实例化）
        if share_weight:
            self.model_2 = self.model_1
        else:
            if use_swin:
                bb2 = models.swin_t(pretrained=True)
                bb2.head = nn.Identity()
                self.model_2 = bb2

            elif use_resnext:
                resx2 = models.resnext50_32x4d(pretrained=True)
                modules2 = list(resx2.children())[:-2]
                self.model_2 = nn.Sequential(*modules2)

            else:
                res2 = models.resnet101(pretrained=True)
                self.model_2 = nn.Sequential(
                    res2.conv1, res2.bn1, res2.relu, res2.maxpool,
                    res2.layer1, res2.layer2, res2.layer3, res2.layer4
                )

        # 3) 池化设置
        if pool == 'avg+max':
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.maxpool = nn.AdaptiveMaxPool2d(1)
            feat_dim *= 2
        elif pool == 'avg':
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        elif pool == 'max':
            self.maxpool = nn.AdaptiveMaxPool2d(1)
        elif pool == 'gem':
            self.gem = GeM(dim=feat_dim)
        else:
            raise ValueError(f"Unsupported pool type: {pool}")

        # 4) 分类头（可选 circle loss 输出特征）
        self.classifier = ClassBlock(
            input_dim=feat_dim,
            class_num=class_num,
            droprate=droprate,
            return_f=circle
        )

    def extract_feats(self, x: torch.Tensor, model: nn.Module) -> torch.Tensor:
        with torch.no_grad():
            fmap = model(x)  # [B,C,H,W]
            if self.pool == 'avg+max':
                a = self.avgpool(fmap); m = self.maxpool(fmap)
                feat = torch.cat([a, m], dim=1)
            elif self.pool == 'avg':
                feat = self.avgpool(fmap)
            elif self.pool == 'max':
                feat = self.maxpool(fmap)
            else:  # gem
                feat = self.gem(fmap)
            feat = feat.flatten(1)
            return F.normalize(feat, dim=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        def _forward_one(x, model):
            fmap = model(x)
            if self.pool == 'avg+max':
                a = self.avgpool(fmap); m = self.maxpool(fmap)
                f = torch.cat([a, m], dim=1)
            elif self.pool == 'avg':
                f = self.avgpool(fmap)
            elif self.pool == 'max':
                f = self.maxpool(fmap)
            else:
                f = self.gem(fmap)
            f = f.flatten(1)
            return self.classifier(f)

        y1 = _forward_one(x1, self.model_1) if x1 is not None else None
        y2 = _forward_one(x2, self.model_2) if x2 is not None else None
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
    # 测试：使用 ResNeXt50_32x4d backbone
    net = two_view_net(
        class_num=751,
        droprate=0.5,
        pool='avg',
        share_weight=False,
        circle=False,
        use_swin=True,
        use_resnext=False
    )
    print(net)
    dummy = Variable(torch.randn(8, 3, 256, 256))
    out1, out2 = net(dummy, dummy)
    print('net output size:', out1.shape, out2.shape)  # 应为 [8, 751]