import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F

######################################################################

class ft_net_swinT(nn.Module):
    def __init__(self, class_num, droprate=0.5, pool='avg'):
        super(ft_net_swinT, self).__init__()
        # 1) 主干 Swin‑Tiny
        model_ft = models.swin_t(pretrained=True)
        # 2) 删除原分类头
        model_ft.head = nn.Identity()

        self.pool = pool
        # Swin‑Tiny 最后一层输出通道是 768
        feat_dim = 768
        if pool == 'avg+max':
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.maxpool = nn.AdaptiveMaxPool2d(1)
            feat_dim *= 2
        elif pool == 'avg':
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        elif pool == 'max':
            self.maxpool = nn.AdaptiveMaxPool2d(1)
        elif pool == 'gem':
            self.gem = GeM(dim=768)

        # 3) 拆分 feature extractor
        self.backbone = model_ft
        self.classifier = ClassBlock(feat_dim, class_num, droprate)

    def forward(self, x):
        # swin_t 的 forward 本身会 return 一个 [B, C, H, W] 特征图
        x = self.backbone(x)       # -> [B,768,H’,W’]
        if self.pool == 'avg+max':
            x = torch.cat([self.avgpool(x), self.maxpool(x)], dim=1)
        elif self.pool == 'avg':
            x = self.avgpool(x)
        elif self.pool == 'max':
            x = self.maxpool(x)
        elif self.pool == 'gem':
            x = self.gem(x)
        x = x.flatten(1)
        return self.classifier(x)

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
            return x, f
        else:
            x = self.classifier(x)
            return x


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
            model_ft.gem2 = GeM(dim = 2048)

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

class two_view_net(nn.Module):
    """
    Two‑view matching network supporting ResNet101 or Swin‑Tiny backbones.
    Args:
        class_num (int): number of classes (for the ClassBlock head).
        droprate (float): dropout rate in the ClassBlock.
        pool (str): one of 'avg', 'max', 'avg+max', or 'gem'.
        share_weight (bool): if True, both views share the same backbone weights.
        circle (bool): whether ClassBlock returns features for circle loss.
        use_swin (bool): if True, use Swin‑Tiny; otherwise ResNet101.
    """
    def __init__(self,
                 class_num: int,
                 droprate: float = 0.5,
                 pool: str = 'avg',
                 share_weight: bool = False,
                 circle: bool = False,
                 use_swin: bool = False):
        super(two_view_net, self).__init__()
        self.use_swin = use_swin
        self.pool     = pool
        self.circle   = circle

        # 1) Backbone selection
        if use_swin:
            # Swin‑Tiny backbone
            swin = models.swin_t(pretrained=True)
            swin.head = nn.Identity()      # remove classifier head
            self.model_1 = swin
            feat_dim = 768                 # Swin‑Tiny output channels
        else:
            # ResNet101 backbone
            res101 = models.resnet101(pretrained=True)
            # optionally adjust stride here if needed
            self.model_1 = nn.Sequential(
                res101.conv1, res101.bn1, res101.relu, res101.maxpool,
                res101.layer1, res101.layer2, res101.layer3, res101.layer4
            )
            feat_dim = 2048                # ResNet101 output channels

        # 2) Second view: share or separate
        if share_weight:
            self.model_2 = self.model_1
        else:
            if use_swin:
                swin2 = models.swin_t(pretrained=True)
                swin2.head = nn.Identity()
                self.model_2 = swin2
            else:
                res101b = models.resnet101(pretrained=True)
                self.model_2 = nn.Sequential(
                    res101b.conv1, res101b.bn1, res101b.relu, res101b.maxpool,
                    res101b.layer1, res101b.layer2, res101b.layer3, res101b.layer4
                )

        # 3) Pooling layer(s)
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

        # 4) Classification head
        self.classifier = ClassBlock(
            input_dim=feat_dim,
            class_num=class_num,
            droprate=droprate,
            return_f=circle
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        def extract_feats(x, model):
            # model(x) for Swin returns [B, C, H, W]; for ResNet we wrapped layers similarly
            feat = model(x)
            if self.pool == 'avg+max':
                feat = torch.cat([self.avgpool(feat), self.maxpool(feat)], dim=1)
            elif self.pool == 'avg':
                feat = self.avgpool(feat)
            elif self.pool == 'max':
                feat = self.maxpool(feat)
            elif self.pool == 'gem':
                feat = self.gem(feat)
            return feat.flatten(1)  # shape [B, feat_dim]

        y1 = None
        y2 = None
        if x1 is not None:
            f1 = extract_feats(x1, self.model_1)
            y1 = self.classifier(f1)
        if x2 is not None:
            f2 = extract_feats(x2, self.model_2)
            y2 = self.classifier(f2)
        return y1, y2

# model.py（在现有代码中添加以下类）

class ft_net_dense(nn.Module):
    """基于DenseNet121的单分支模型，适配原项目池化方式"""
    def __init__(self, class_num, droprate=0.5, stride=2, pool='avg', init_model=None):
        super(ft_net_dense, self).__init__()
        
        # 1. 加载预训练DenseNet121并移除原分类器
        model_ft = models.densenet121(pretrained=True)
        model_ft.classifier = nn.Sequential()  # 删除原分类层
        
        # 2. 配置池化层
        self.pool = pool
        if pool == 'avg+max':
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.maxpool = nn.AdaptiveMaxPool2d((1,1))
            input_dim = 1024 * 2  # 特征拼接后维度
        elif pool == 'avg':
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            input_dim = 1024
        elif pool == 'max':
            self.maxpool = nn.AdaptiveMaxPool2d((1,1))
            input_dim = 1024
        elif pool == 'gem':
            self.gem = GeM(dim=1024)  # DenseNet特征维度为1024
            input_dim = 1024
        else:
            raise ValueError(f"Unsupported pool type: {pool}")
        
        # 3. 特征提取主干网络
        self.features = model_ft.features
        
        # 4. 分类器（动态调整输入维度）
        self.classifier = ClassBlock(
            input_dim=input_dim, 
            class_num=class_num,
            droprate=droprate,
            linear=True,
            return_f=False
        )

    def forward(self, x):
        # 特征提取
        x = self.features(x)
        
        # 池化处理
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
        
        # 展平维度
        x = x.view(x.size(0), -1)
        
        # 分类输出
        x = self.classifier(x)
        return x
        
class three_view_net(nn.Module):
    def __init__(self, class_num, droprate, stride=2, pool='avg', share_weight=False, circle=False, use_dense=False):
        super(three_view_net, self).__init__()
        if use_dense:
            # 使用DenseNet121并提取特征部分
            self.model_1 = models.densenet121(pretrained=True).features
            self.model_2 = models.densenet121(pretrained=True).features
            self.model_3 = models.densenet121(pretrained=True).features
            input_dim = 1024
        else:
            # 使用ResNet101并调整池化层
            self.model_1 = models.resnet101(pretrained=True)
            self.model_2 = models.resnet101(pretrained=True)
            self.model_3 = models.resnet101(pretrained=True)
            # 修改ResNet的池化层
            if pool == 'gem':
                self.model_1.avgpool = GeM(dim=2048)
                self.model_2.avgpool = GeM(dim=2048)
                self.model_3.avgpool = GeM(dim=2048)
            else:
                self.model_1.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.model_2.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.model_3.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            input_dim = 2048

        self.circle = circle
        self.use_dense = use_dense
        # 确保分类器输入维度正确
        self.classifier = ClassBlock(input_dim, class_num, droprate, return_f=circle)

    def forward(self, x1, x2, x3, x4=None):
        def process_input(x, model):
            if self.use_dense:
                # DenseNet特征处理
                features = model(x)
                features = F.adaptive_avg_pool2d(features, (1, 1))
                features = torch.flatten(features, 1)
            else:
                # ResNet前向传播
                x = model.conv1(x)
                x = model.bn1(x)
                x = model.relu(x)
                x = model.maxpool(x)
                x = model.layer1(x)
                x = model.layer2(x)
                x = model.layer3(x)
                x = model.layer4(x)
                x = model.avgpool(x)
                features = torch.flatten(x, 1)
            return features

        if x1 is not None:
            features1 = process_input(x1, self.model_1)
            y1 = self.classifier(features1)
        else:
            y1 = None

        if x2 is not None:
            features2 = process_input(x2, self.model_2)
            y2 = self.classifier(features2)
        else:
            y2 = None

        if x3 is not None:
            features3 = process_input(x3, self.model_3)
            y3 = self.classifier(features3)
        else:
            y3 = None

        if x4 is not None:
            features4 = process_input(x4, self.model_2)
            y4 = self.classifier(features4)
            return y1, y2, y3, y4
        else:
            return y1, y2, y3

'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
    # 示例 1：使用 Swin‑Tiny 主干
    net = two_view_net(751, droprate=0.5, use_swin=True)
    # 示例 2：使用默认 ResNet101 主干
    # net = two_view_net(751, droprate=0.5)

    print(net)
    # 随机输入一个 batch 大小为 8 的 3×256×256 图像
    input_tensor = Variable(torch.randn(8, 3, 256, 256))

    # forward
    out1, out2 = net(input_tensor, input_tensor)
    print('net output size:')
    print(out1.shape, out2.shape)   # 都应该是 [8, 751]
