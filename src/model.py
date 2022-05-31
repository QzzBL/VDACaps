from torch import Tensor
import  torch
from    torch import  nn
from    torch.nn import functional as F
import math
from torch.autograd import Variable


NUM_CLASSES = 7
NUM_ROUTING_ITERATIONS = 3

#===============DA===============#
class k3_dilation3_SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(k3_dilation3_SpatialAttention, self).__init__()
        self.kernel_size = kernel_size
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=3, bias=False, dilation=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
#===============DA===============#

#===============DSA(kernel size:3、7)===============#
class k3_DSA(nn.Module):
    def __init__(self, kernel_size=3):
        super(k3_DSA, self).__init__()
        self.kernel_size = kernel_size
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class k7_DSA(nn.Module):
    def __init__(self, kernel_size=7):
        super(k7_DSA, self).__init__()
        self.kernel_size=kernel_size
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
#===============DSA(kernel size:3、7)===============#


#===============channel_shuffle===============#
def channel_shuffle(x: Tensor, groups: int) -> Tensor:

    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x
#===============channel_shuffle===============#


#===============SeparableConv2d===============#
class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, wh):
        super(SeparableConv2d, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=wh,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out
#===============SeparableConv2d===============#


#==========================VSSCaps==========================#
class VSSCaps(nn.Module):
    """Constructs VSSCaps module.
    """

    def __init__(self, inplanes, groups):
        super(VSSCaps, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2,bias=False)  #(FOLCIC)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=5, padding=(5 - 1) // 2, bias=False) #(FOLCIC)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=7, padding=(7 - 1) // 2, bias=False) #(FOLCIC)
        self.conv4 = nn.Conv1d(1, 1, kernel_size=9, padding=(9 - 1) // 2, bias=False) #(FOLCIC)

        #Depthwise seperable convolution
        self.separableConv2d = SeparableConv2d(in_ch=8, out_ch=8, wh=6)

        #DSA attention
        self.k3_dsa = k3_DSA()
        self.k7_dsa = k7_DSA()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        b, c, h, w = x.shape

        # groups = 64
        x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25, x_26, x_27, x_28, x_29, x_30, x_31, \
        x_32, x_33, x_34, x_35, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53, x_54, x_55, x_56, x_57, x_58, x_59, x_60, x_61, x_62, x_63 = x.chunk(
            64, dim=1)

        x_chunk_channel = x_0

        x_chunk = [x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17,
                   x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25, x_26, x_27, x_28, x_29, x_30, x_31, \
                   x_32, x_33, x_34, x_35, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48,
                   x_49, x_50, x_51, x_52, x_53, x_54, x_55, x_56, x_57, x_58, x_59, x_60, x_61, x_62, x_63]


        j = 0 # serial number
        pre_VCA_kernel_size = ''
        pre_DSA_kernel_size = ''

        for i in x_chunk:
            z = i

            #Depthwise seperable convolution
            i = self.separableConv2d(i)

            #The convolution kernel sizes of 32 VCA sequences are 3、5、7 respectively
            if j % 2 == 0:
                if pre_VCA_kernel_size == '':
                    i = self.conv(i.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
                    attention_weights = self.sigmoid(i)
                    features = x_chunk_channel * attention_weights.expand_as(x_chunk_channel)
                    pre_VCA_kernel_size = '3'
                elif pre_VCA_kernel_size == '3':
                    i = self.conv2(i.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
                    attention_weights = self.sigmoid(i)
                    features = x_chunk_channel * attention_weights.expand_as(x_chunk_channel)
                    pre_VCA_kernel_size = '5'
                elif pre_VCA_kernel_size == '5':
                    i = self.conv3(i.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
                    attention_weights = self.sigmoid(i)
                    features = x_chunk_channel * attention_weights.expand_as(x_chunk_channel)
                    pre_VCA_kernel_size = '7'
                elif pre_VCA_kernel_size == '7':
                    i = self.conv(i.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
                    attention_weights = self.sigmoid(i)
                    features = x_chunk_channel * attention_weights.expand_as(x_chunk_channel)
                    pre_VCA_kernel_size = '3'

                j = j + 1
            else:
                # The convolution kernel sizes of 32 DSA sequences are 3、7 respectively
                if pre_DSA_kernel_size == '':
                    z = self.k3_dsa(z) * z
                    pre_DSA_kernel_size = '3'
                elif pre_DSA_kernel_size == '3':
                    z = self.k7_dsa(z) * z
                    pre_DSA_kernel_size = '7'
                elif pre_DSA_kernel_size == '7':
                    z = self.k3_dsa(z) * z
                    pre_DSA_kernel_size = '3'

                j = j + 1

        out = torch.cat(
            [x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18,
             x_19, x_20, x_21, x_22, x_23, x_24, x_25, x_26, x_27, x_28, x_29, x_30, x_31, \
             x_32, x_33, x_34, x_35, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49,
             x_50, x_51, x_52, x_53, x_54, x_55, x_56, x_57, x_58, x_59, x_60, x_61, x_62, x_63], dim=1)

        return out
#==========================VSSCaps_End==========================#


#==========================VCA_DA_Layer==========================#
class VCA_DA_layer(nn.Module):
    """Constructs a VCA_DA module.

    Args:
        channel: Number of channels of the input feature map
    """
    def __init__(self, channel, k_size=3):
        super(VCA_DA_layer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

        #dilated spatial attention (DA atention)
        self.spatial_DA = k3_dilation3_SpatialAttention()

        #Depthwise seperable convolution
        self.separableConv2d_56 = SeparableConv2d(in_ch=channel, out_ch=channel, wh=56)
        self.separableConv2d_28 = SeparableConv2d(in_ch=channel, out_ch=channel, wh=28)
        self.separableConv2d_14 = SeparableConv2d(in_ch=channel, out_ch=channel, wh=14)
        self.separableConv2d_7 = SeparableConv2d(in_ch=channel, out_ch=channel, wh=7)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()
        z = x
        s = x

        #‘s’ is the output of the DA
        s = self.spatial_DA(s) * s

        #Different feature map sizes correspond to different Depthwise seperable convolution
        if h == 56:
            y = self.separableConv2d_56(x)
        elif h == 28:
            y = self.separableConv2d_28(x)
        elif h == 14:
            y = self.separableConv2d_14(x)
        else :
            y = self.separableConv2d_7(x)

        #fast one-dimensional local cross-channel interaction convolution (FOLCIC)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # VCA + DA
        features = y + s

        attention_weights = self.sigmoid(features)

        return z * attention_weights.expand_as(z)

#==========================VCA_DA_End==========================#


#==========================ResNet==========================#

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.vca_da = VCA_DA_layer(planes, k_size)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.vca_da(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.vca_da = VCA_DA_layer(planes, k_size)
        self.downsample = downsample
        self.stride = stride

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
        out = self.vca_da(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=7, k_size=[3, 3, 3, 3]):

        self.inplanes = 64
        super(ResNet, self).__init__()

        self.conv = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], int(k_size[0]), stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], int(k_size[1]), stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], int(k_size[2]), stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], int(k_size[3]), stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(256 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, k_size, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, k_size))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, k_size=k_size))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet18(k_size=[3, 3, 3, 3], num_classes=7, pretrained=False):
    """Constructs a ResNet-18 model.

    Args:
        k_size: Adaptive selection of kernel size
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes:The classes of classification
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, k_size=k_size)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def resnet34(k_size=[3, 3, 3, 3], num_classes=7, pretrained=False):
    """Constructs a ResNet-34 model.

    Args:
        k_size: Adaptive selection of kernel size
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes:The classes of classification
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, k_size=k_size)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model

#==========================ResNet_End==========================#


def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)

class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=NUM_ROUTING_ITERATIONS):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations
        self.num_capsules = num_capsules
        self.vsscaps = VSSCaps(out_channels,groups=32)

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            self.conv2 = nn.Conv2d(in_channels, out_channels * num_capsules, kernel_size=kernel_size, stride=stride, padding=0)

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)


    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]
            logits = Variable(torch.zeros(*priors.size()))
            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))
                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = self.conv2(x)

            # Part II of VDACaps
            outputs = self.vsscaps(outputs)
            outputs = channel_shuffle(outputs,groups=64)
            outputs = outputs.view(x.size(0), -1, 8)
            outputs = self.squash(outputs)

        return outputs

class VDACaps(nn.Module):
    def __init__(self):
        super(VDACaps, self).__init__()

        self.resnet18=resnet18()
        self.conv2_VSSCaps = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=512, out_channels=64,
                                             kernel_size=2, stride=1)
        self.expression_capsules = CapsuleLayer(num_capsules=NUM_CLASSES, num_route_nodes=64 * 6 * 6, in_channels=8,
                                           out_channels=16)
        self.decoder = nn.Sequential(
            nn.Linear(16 * NUM_CLASSES, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 150528),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):

        x = self.resnet18(x)
        x = self.conv2_VSSCaps(x)
        t = self.expression_capsules(x)
        x = t.squeeze().transpose(0, 1)

        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)
        _, max_length_indices = classes.max(dim=1)

        masked = Variable(torch.eye(NUM_CLASSES)).index_select(dim=0, index=max_length_indices)

        reconstructions = self.decoder((x * masked[:, :, None]).contiguous().view(x.size(0), -1))

        return masked, classes, reconstructions


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss()

    def forward(self, images, labels, classes, reconstructions):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()
        images = images.view(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)