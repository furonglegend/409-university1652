import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F

######################################################################
class GeM(nn.Module):
    def __init__(self, dim=2048, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(dim)*p, requires_grad=True)
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p, eps):
        x = torch.transpose(x, 1, -1)
        x = x.clamp(min=eps).pow(p)
        x = torch.transpose(x, 1, -1)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        x = x.view(x.size(0), x.size(1))
        x = x.pow(1./p)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.mean():.4f}, eps={self.eps}, dim={self.dim})"

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
        self.add_block = nn.Sequential(*add_block)
        self.add_block.apply(weights_init_kaiming)
        
        self.classifier = nn.Linear(num_bottleneck, class_num)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            return self.classifier(x), x
        else:
            return self.classifier(x)
# 在model.py中添加以下类定义
class three_view_net(nn.Module):
    def __init__(self, class_num, droprate, stride=2, pool='avg', share_weight=False, model_type='efficient', circle=False):
        super(three_view_net, self).__init__()
        # Select backbone model
        if model_type == 'efficient':
            model_creator = ft_net_Efficient
            feat_dim = 1792 * (2 if 'avg+max' in pool else 1)
        elif model_type == 'convnext':
            model_creator = ft_net_ConvNeXt
            feat_dim = 1536 * (2 if 'avg+max' in pool else 1)
        else:
            raise ValueError("Unsupported model type")
        
        # Initialize backbone models
        self.model_1 = model_creator(class_num, stride=stride, pool=pool)
        self.model_2 = model_creator(class_num, stride=stride, pool=pool)
        self.model_3 = self.model_1 if share_weight else model_creator(class_num, stride=stride, pool=pool)
        
        # Initialize classifier
        self.classifier = ClassBlock(feat_dim, class_num, droprate, return_f=circle)

    def forward(self, x1, x2, x3, x4=None):
        out1 = self.classifier(self.model_1(x1)) if x1 is not None else None
        out2 = self.classifier(self.model_2(x2)) if x2 is not None else None
        out3 = self.classifier(self.model_3(x3)) if x3 is not None else None
        
        if x4 is not None:
            out4 = self.classifier(self.model_2(x4))  # 共享第二个模型的权重
            return out1, out2, out3, out4
        return out1, out2, out3
        
class ft_net_Efficient(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net_Efficient, self).__init__()
        model_ft = models.efficientnet_b4(pretrained=True)
        self.features = model_ft.features
        self.pool = pool
        self.feat_dim = 1792  # EfficientNet-B4 feature dimension
        
        # Initialize pooling layers
        if 'avg' in pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        if 'max' in pool:
            self.maxpool = nn.AdaptiveMaxPool2d((1,1))
        if 'gem' in pool:
            self.gem = GeM(dim=self.feat_dim)

    def forward(self, x):
        x = self.features(x)
        if self.pool == 'avg+max':
            x1 = self.avgpool(x)
            x2 = self.maxpool(x)
            x = torch.cat((x1, x2), dim=1)
        elif self.pool == 'avg':
            x = self.avgpool(x)
        elif self.pool == 'max':
            x = self.maxpool(x)
        elif self.pool == 'gem':
            x = self.gem(x)
        x = x.view(x.size(0), -1)
        return x

class ft_net_ConvNeXt(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net_ConvNeXt, self).__init__()
        model_ft = models.convnext_large(pretrained=True)
        self.features = model_ft.features
        self.pool = pool
        self.feat_dim = 1536  # ConvNeXt-Large feature dimension
        
        # Initialize pooling layers
        if 'avg' in pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        if 'max' in pool:
            self.maxpool = nn.AdaptiveMaxPool2d((1,1))
        if 'gem' in pool:
            self.gem = GeM(dim=self.feat_dim)

    def forward(self, x):
        x = self.features(x)
        if self.pool == 'avg+max':
            x1 = self.avgpool(x)
            x2 = self.maxpool(x)
            x = torch.cat((x1, x2), dim=1)
        elif self.pool == 'avg':
            x = self.avgpool(x)
        elif self.pool == 'max':
            x = self.maxpool(x)
        elif self.pool == 'gem':
            x = self.gem(x)
        x = x.view(x.size(0), -1)
        return x

class two_view_net(nn.Module):
    def __init__(self, class_num, droprate, stride=2, pool='avg', share_weight=False, model_type='efficient', circle=False):
        super(two_view_net, self).__init__()
        # Select backbone model
        if model_type == 'efficient':
            model_creator = ft_net_Efficient
            feat_dim = 1792 * (2 if 'avg+max' in pool else 1)
        elif model_type == 'convnext':
            model_creator = ft_net_ConvNeXt
            feat_dim = 1536 * (2 if 'avg+max' in pool else 1)
        else:
            raise ValueError("Unsupported model type")
        
        # Initialize backbone models
        self.model_1 = model_creator(class_num, stride=stride, pool=pool)
        self.model_2 = self.model_1 if share_weight else model_creator(class_num, stride=stride, pool=pool)
        
        # Initialize classifier
        self.classifier = ClassBlock(feat_dim, class_num, droprate, return_f=circle)

    def forward(self, x1, x2):
        out1 = self.classifier(self.model_1(x1)) if x1 is not None else None
        out2 = self.classifier(self.model_2(x2)) if x2 is not None else None
        return out1, out2

# Testing
if __name__ == '__main__':
    # Test EfficientNet-based model
    net = two_view_net(751, droprate=0.5, pool='gem', model_type='efficient')
    input = torch.randn(8, 3, 256, 256)
    output1, output2 = net(input, input)
    print('EfficientNet output shape:', output1[0].shape if net.classifier.return_f else output1.shape)
    
    # Test ConvNeXt-based model
    net = two_view_net(751, droprate=0.5, pool='avg+max', model_type='convnext')
    output1, output2 = net(input, input)
    print('ConvNeXt output shape:', output1[0].shape if net.classifier.return_f else output1.shape)