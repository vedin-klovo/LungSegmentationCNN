import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class SimpleConvolution(nn.Module):
    def __init__(self, input_channels, output_channels, momentum=0.5):
        super(SimpleConvolution, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, (3, 3), padding=1)
        self.batch_norm = nn.BatchNorm2d(output_channels, momentum=momentum)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.ReLU(x)

        return x


class Convolution(nn.Module):
    def __init__(self, input_channels, output_channels, momentum=0.5):
        super(Convolution, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, (3, 3), padding=1)
        self.batch_norm = nn.BatchNorm2d(output_channels, momentum=momentum)
        self.ReLU = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2, stride=2, return_indices=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.ReLU(x)
        x, idc1 = self.max_pool(x)

        return x, idc1


class UpSample(nn.Module):
    def __init__(self, kernel_size, stride=2):
        super(UpSample, self).__init__()
        self.max_unpool = nn.MaxUnpool2d(kernel_size, stride=stride)

    def forward(self, x, idc, output_size=None):
        x = self.max_unpool(x, idc, output_size=output_size)

        return x


class SegNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(SegNet, self).__init__()

        self.downConv0 = SimpleConvolution(input_channels, 64)
        self.downConv1 = Convolution(64, 64)

        self.downConv2 = SimpleConvolution(64, 128)
        self.downConv3 = Convolution(128, 128)

        self.downConv4 = SimpleConvolution(128, 256)
        self.downConv5 = SimpleConvolution(256, 256)
        self.downConv6 = Convolution(256, 256)

        self.downConv7 = SimpleConvolution(256, 512)
        self.downConv8 = SimpleConvolution(512, 512)
        self.downConv9 = Convolution(512, 512)

        self.downConv10 = SimpleConvolution(512, 512)
        self.downConv11 = SimpleConvolution(512, 512)
        self.downConv12 = Convolution(512, 512)

        self.upSample = UpSample(2, 2)

        self.upConv12 = SimpleConvolution(512, 512)
        self.upConv11 = SimpleConvolution(512, 512)
        self.upConv10 = SimpleConvolution(512, 512)

        self.upConv9 = SimpleConvolution(512, 512)
        self.upConv8 = SimpleConvolution(512, 512)
        self.upConv7 = SimpleConvolution(512, 256)

        self.upConv6 = SimpleConvolution(256, 256)
        self.upConv5 = SimpleConvolution(256, 256)
        self.upConv4 = SimpleConvolution(256, 128)

        self.upConv3 = SimpleConvolution(128, 128)
        self.upConv2 = SimpleConvolution(128, 64)

        self.upConv1 = SimpleConvolution(64, 64)
        self.upConv0 = SimpleConvolution(64, num_classes)

    def forward(self, x):
        x = self.downConv0(x)
        x, idc1 = self.downConv1(x)
        size1 = x.size()

        x = self.downConv2(x)
        x, idc2 = self.downConv3(x)
        size2 = x.size()

        x = self.downConv4(x)
        x = self.downConv5(x)
        x, idc3 = self.downConv6(x)
        size3 = x.size()

        x = self.downConv7(x)
        x = self.downConv8(x)
        x, idc4 = self.downConv9(x)
        size4 = x.size()

        x = self.downConv10(x)
        x = self.downConv11(x)
        x, idc5 = self.downConv12(x)

        x = self.upSample(x, idc5, output_size=size4)
        x = self.upConv12(x)
        x = self.upConv11(x)
        x = self.upConv10(x)

        x = self.upSample(x, idc4, output_size=size3)
        x = self.upConv9(x)
        x = self.upConv8(x)
        x = self.upConv7(x)

        x = self.upSample(x, idc3, output_size=size2)
        x = self.upConv6(x)
        x = self.upConv5(x)
        x = self.upConv4(x)

        x = self.upSample(x, idc2, output_size=size1)
        x = self.upConv3(x)
        x = self.upConv2(x)

        x = self.upSample(x, idc1)
        x = self.upConv1(x)
        x = self.upConv0(x)

        return x


if __name__ == '__main__':
    torch.no_grad()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    segnet = SegNet(1, 2).cuda()

    img = Image.open("../data/Montgomery/images/MCUCXR_0001_0.png")
    img_resized = img.resize((572, 572))

    convert_tensor = transforms.ToTensor()
    inp = convert_tensor(img_resized).to(device).unsqueeze(0)
    out = segnet(inp)
    probs = torch.sigmoid(out)[0]
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img.size),
        transforms.ToTensor()
    ])

    full_mask = tf(probs.cpu()).squeeze()

    mask_np = (full_mask > 0.5).numpy()
    im_out = Image.fromarray((np.swapaxes(mask_np, 0, 2) * 255).astype(np.uint8))
    im_out.save("test.png")
