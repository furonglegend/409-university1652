import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F

class Projection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Projection, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.fc(x)


class ModelAveraging(nn.Module):
    def __init__(self, models):
        super(ModelAveraging, self).__init__()
        # 使用 ModuleList 保存不同的基础模型
        self.models = nn.ModuleList(models)
        self.n_models = len(models)
        # 初始化权重为均匀分布，可学习。通过 softmax 保证归一化
        self.weights = nn.Parameter(torch.ones(self.n_models) / self.n_models)

    def forward(self, x):
        # 分别获取每个模型的输出，要求每个模型的输出维度一致（例如 [batch_size, output_dim]）
        outputs = [model(x) for model in self.models]

        # 对权重进行 softmax 归一化，确保和为1
        normalized_weights = torch.softmax(self.weights, dim=0)

        # 计算加权平均。利用列表解析，每个模型的输出乘以对应的权重后累加
        weighted_outputs = [w * output for w, output in zip(normalized_weights, outputs)]
        averaged_output = sum(weighted_outputs)
        return averaged_output


# Define DenseNet-based model
class ft_net_DenseNet(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net_DenseNet, self).__init__()
        model_ft = models.densenet121(pretrained=True)  # DenseNet model
        self.pool = pool
        if pool == 'gem':
            model_ft.gem2 = GeM(dim=1024)

        self.model = model_ft

        if init_model is not None:
            self.model = init_model.model
            self.pool = init_model.pool

    def forward(self, x):
        x = self.model.features(x)
        if self.pool == 'gem':
            x = self.model.gem2(x)
        x = x.view(x.size(0), x.size(1))
        return x

# Define Inception-based model
class ft_net_Inception(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net_Inception, self).__init__()
        model_ft = models.inception_v3(pretrained=True)  # Inception model
        self.pool = pool
        if pool == 'gem':
            model_ft.gem2 = GeM(dim=2048)

        self.model = model_ft

        if init_model is not None:
            self.model = init_model.model
            self.pool = init_model.pool

    def forward(self, x):
        x = self.model.Conv2d_1a_3x3(x)
        x = self.model.Conv2d_2a_3x3(x)
        x = self.model.Conv2d_2b_3x3(x)
        x = self.model.maxpool1(x)
        x = self.model.Conv2d_3b_1x1(x)
        x = self.model.Conv2d_4a_3x3(x)
        x = self.model.maxpool2(x)
        x = self.model.Mixed_5b(x)
        x = self.model.Mixed_5c(x)
        x = self.model.Mixed_5d(x)
        x = self.model.Mixed_6a(x)
        x = self.model.Mixed_6b(x)
        x = self.model.Mixed_6c(x)
        x = self.model.Mixed_6d(x)
        x = self.model.Mixed_6e(x)
        x = self.model.Mixed_7a(x)
        x = self.model.Mixed_7b(x)
        x = self.model.Mixed_7c(x)
        
        if self.pool == 'gem':
            x = self.model.gem2(x)
        x = x.view(x.size(0), x.size(1))
        return x

######################################################################
class GeM(nn.Module):
    def __init__(self, dim=2048, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(dim) * p, requires_grad=True)  # initial p
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
        self.pool = pool
        if pool == 'gem':
            model_ft.gem2 = GeM(dim=512)

        self.model = model_ft

        if init_model is not None:
            self.model = init_model.model
            self.pool = init_model.pool

    def forward(self, x):
        x = self.model.features(x)
        if self.pool == 'gem':
            x = self.model.gem2(x)

        x = x.view(x.size(0), x.size(1))
        return x


# Define the ResNet50-based Model
# Define the ResNet50-based Model
class ft_net(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        if pool == 'gem':
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
        if self.pool == 'gem':
            x = self.model.gem2(x)

        x = x.view(x.size(0), x.size(1))
        return x
        
class two_view_net(nn.Module):
    def __init__(self, class_num, droprate, stride=2, pool='avg', share_weight=False,
                 VGG16=False, ResNet50=False, DenseNet=False, Inception=False, circle=False):
        super(two_view_net, self).__init__()
        self.target_dim = 2048  # 统一目标维度
        
        # 选择模型时加入投影
        if VGG16:
            m1 = ft_net_VGG16(class_num, stride=stride, pool=pool)  # output 512
            proj1 = Projection(512, self.target_dim)
            self.model_1 = nn.Sequential(m1, proj1)
        elif ResNet50:
            self.model_1 = ft_net(class_num, stride=stride, pool=pool)  # 输出 2048
        elif DenseNet:
            m1 = ft_net_DenseNet(class_num, stride=stride, pool=pool)  # 输出 1024
            proj1 = Projection(1024, self.target_dim)
            self.model_1 = nn.Sequential(m1, proj1)
        elif Inception:
            self.model_1 = ft_net_Inception(class_num, stride=stride, pool=pool)  # 输出 2048

        if share_weight:
            self.model_2 = self.model_1
        else:
            # 同理构造第二分支
            if VGG16:
                m2 = ft_net_VGG16(class_num, stride=stride, pool=pool)
                proj2 = Projection(512, self.target_dim)
                self.model_2 = nn.Sequential(m2, proj2)
            elif ResNet50:
                self.model_2 = ft_net(class_num, stride=stride, pool=pool)
            elif DenseNet:
                m2 = ft_net_DenseNet(class_num, stride=stride, pool=pool)
                proj2 = Projection(1024, self.target_dim)
                self.model_2 = nn.Sequential(m2, proj2)
            elif Inception:
                self.model_2 = ft_net_Inception(class_num, stride=stride, pool=pool)
        
        # classifier 输入调整为 target_dim
        self.circle = circle
        self.classifier = ClassBlock(self.target_dim, class_num, droprate, return_f=circle)
        if pool == 'avg+max':
            self.classifier = ClassBlock(self.target_dim*2, class_num, droprate, return_f=circle)

    def forward(self, x1, x2):
        f1 = self.model_1(x1)
        y1 = self.classifier(f1)
        f2 = self.model_2(x2)
        y2 = self.classifier(f2)
        return y1, y2


# Modify three_view_net to support model averaging or stacking
class three_view_net(nn.Module):
    def __init__(self, class_num, droprate, stride=2, pool='avg', share_weight=False,
                 VGG16=False, ResNet50=False, DenseNet=False, Inception=False, circle=False):
        super().__init__()
        self.target_dim = 2048
        self.models_list = nn.ModuleList()

        # 动态添加模型并统一维度
        def add_model(model_class, in_dim):
            model = model_class(class_num, droprate, stride, pool=pool)
            if in_dim != self.target_dim:
                return nn.Sequential(model, Projection(in_dim, self.target_dim))
            return model

        if VGG16:
            self.models_list.append(add_model(ft_net_VGG16, 512))  # VGG输出512维
        if ResNet50:
            self.models_list.append(add_model(ft_net, 2048))      # ResNet输出2048维
        if DenseNet:
            self.models_list.append(add_model(ft_net_DenseNet, 1024)) # DenseNet1024维
        if Inception:
            self.models_list.append(add_model(ft_net_Inception, 2048))
        
        # 确保模型列表不为空
        assert len(self.models_list) > 0, "未选择任何Backbone模型！"
        
        self.model_ensemble = ModelAveraging(self.models_list)
        
        # 动态融合权重
        self.fusion = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
            nn.Softmax(dim=1)
        )
        
        # 分类器
        self.classifier = ClassBlock(2048, class_num, droprate, return_f=circle)

    def forward(self, x1, x2, x3, x4=None):
        # 特征提取
        f1 = self.model_ensemble(x1)
        f2 = self.model_ensemble(x2)
        f3 = self.model_ensemble(x3)
        
        # 类型检查
        assert isinstance(f1, torch.Tensor), "f1应为张量！"
        assert isinstance(f2, torch.Tensor), "f2应为张量！"
        assert isinstance(f3, torch.Tensor), "f3应为张量！"
        
        # 动态融合
        combined = torch.stack([f1, f2, f3], dim=1)  # [batch,3,2048]
        weights = self.fusion(combined.mean(dim=2))  # [batch,3]
        fused = (combined * weights.unsqueeze(2)).sum(dim=1)
        
        return self.classifier(fused)


'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
    # 显式设置circle参数进行测试
    for circle_flag in [False, True]:
        print(f"\n===== Testing circle={circle_flag} =====")
        # 初始化模型（注意添加circle参数）
        net = three_view_net(751, droprate=0.5, 
                            VGG16=True, ResNet50=True, 
                            DenseNet=True, Inception=True,
                            circle=circle_flag)  # 新增参数
        
        # 生成测试输入
        input = torch.randn(8, 3, 256, 256)  # 使用torch代替Variable（新版已整合）
        
        # 前向传播
        outputs = net(input, input, input)
        
        # 处理多输出类型
        if circle_flag:
            logits, features = outputs
            print("Logits shape:", logits.shape)
            print("Features shape:", features.shape)

        else:
            print("Output shape:", outputs.shape)
 
        
        # 打印模型结构
        print("\nModel structure:")
        print(net)