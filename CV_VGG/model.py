import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Dict, Any, cast

cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class vggnet(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.cfg = cfgs[cfg]
        self.in_channels = 3
        self.batch_norm = True
        self.conv_layers = self.create_conv_layers(self.cfg, self.batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.output = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )

        init_weights = True
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        # print(x.shape)
        x = self.output(x)
        return x

    def create_conv_layers(self, architecture, batch_norm: bool = False) -> nn.Sequential:
        layers: List[nn.Module] = []
        in_channels = self.in_channels
        for x in architecture:
            if type(x) == int:
                out_channels = cast(int, x)
                conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = out_channels
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        return nn.Sequential(*layers)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model = vggnet(cfg="A", num_classes=500).to(device)
    print(model)
    # x = torch.randn(1, 3, 224, 224).to(device)
    # print(model(x).shape)
