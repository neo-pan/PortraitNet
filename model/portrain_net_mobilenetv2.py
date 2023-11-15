import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

import numpy as np
        
        
class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation=nn.ReLU):
        super().__init__()
        
        self.left = nn.Sequential(
            self._conv_bn(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
        )
        
        self.right = nn.Sequential(
            self._conv_bn(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels),
            activation(inplace=True),
            self._conv_bn(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            activation(inplace=True),

            self._conv_bn(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels),
            activation(inplace=True),
            self._conv_bn(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
        )
        
        self.activation = activation(inplace=True)

    def _conv_bn(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.right(x) + self.left(x)
        out = self.activation(out)

        return out
    
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        output = self.upsample(x)
        return output
    
class Encoder(nn.Module):
    def __init__(self, args):
        '''
        PortrainNet with MobileNetV2 backbone
        '''
        super().__init__()
        self.args = args

        ''' MobileNetV2 features downsampling scale
        [112, 112]
        [112, 112]
        [56, 56]
        [56, 56]
        [28, 28]
        [28, 28]
        [28, 28]
        [14, 14]
        [14, 14]
        [14, 14]
        [14, 14]
        [14, 14]
        [14, 14]
        [14, 14]
        [7, 7]
        [7, 7]
        [7, 7]
        [7, 7]
        '''
        mobilenet = mobilenet_v2(pretrained=args.pretrained)
        if args.freeze:
            for param in mobilenet.parameters():
                param.requires_grad = False

        self.out_channels = []

        self.down2x = mobilenet.features[0:2]
        self.out_channels.append(mobilenet.features[1].out_channels)

        self.down4x = mobilenet.features[2:4]
        self.out_channels.append(mobilenet.features[3].out_channels)

        self.down8x = mobilenet.features[4:7]
        self.out_channels.append(mobilenet.features[6].out_channels)

        self.down16x = mobilenet.features[7:14]
        self.out_channels.append(mobilenet.features[13].out_channels)

        self.down32x = mobilenet.features[14:18]
        self.out_channels.append(mobilenet.features[17].out_channels)

        del mobilenet
        if not args.pretrained:
            self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        feature2x = self.down2x(x)
        feature4x = self.down4x(feature2x)
        feature8x = self.down8x(feature4x)
        feature16x = self.down16x(feature8x)
        feature32x = self.down32x(feature16x)

        return feature2x, feature4x, feature8x, feature16x, feature32x

class PortrainNetMobileNetV2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_classes = args.num_classes
        
        self.encoder = Encoder(args)
        self.out_channels = self.encoder.out_channels

        self.d_block32x = DBlock(self.out_channels[4], self.out_channels[3])
        self.d_block16x = DBlock(self.out_channels[3], self.out_channels[2])
        self.d_block8x = DBlock(self.out_channels[2], self.out_channels[1])
        self.d_block4x = DBlock(self.out_channels[1], self.out_channels[0])
        self.d_block2x = DBlock(self.out_channels[0], self.out_channels[0])

        self.upsample32x = Upsample(self.out_channels[3], self.out_channels[3])
        self.upsample16x = Upsample(self.out_channels[2], self.out_channels[2])
        self.upsample8x = Upsample(self.out_channels[1], self.out_channels[1])
        self.upsample4x = Upsample(self.out_channels[0], self.out_channels[0])
        self.upsample2x = Upsample(self.out_channels[0], self.out_channels[0])

        self.mask_conv = nn.Conv2d(self.out_channels[0], self.num_classes, kernel_size=3, stride=1, padding=1)
        self.boundary_conv = nn.Conv2d(self.out_channels[0], self.num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        feature2x, feature4x, feature8x, feature16x, feature32x = self.encoder(x)

        up16x = self.upsample32x(self.d_block32x(feature32x))
        up8x = self.upsample16x(self.d_block16x(feature16x + up16x))
        up4x = self.upsample8x(self.d_block8x(feature8x + up8x))
        up2x = self.upsample4x(self.d_block4x(feature4x + up4x))
        up1x = self.upsample2x(self.d_block2x(feature2x + up2x))

        mask_logits = self.mask_conv(up1x)
        boundary_logits = self.boundary_conv(up1x)

        return mask_logits, boundary_logits
        