import torch
import torch.nn as nn
from torch import Tensor
from torchvision import transforms
from PIL import Image


def crop_img(source_tensor: Tensor, target_tensor: Tensor):
    source_tensor_size = source_tensor.size()[2]
    target_tensor_size = target_tensor.size()[2]
    diff = (source_tensor_size - target_tensor_size) // 2
    return source_tensor[:, :, diff:-diff, diff:-diff]


class SimpleConvolution(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(SimpleConvolution, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, (3, 3))
        self.conv2 = nn.Conv2d(output_channel, output_channel, (3, 3))
        self.Relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Relu(x)
        x = self.conv2(x)
        x = self.Relu(x)
        x = self.dropout(x)

        return x


class DownConvolution(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(DownConvolution, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, (3, 3))
        self.conv2 = nn.Conv2d(output_channel, output_channel, (3, 3))
        self.maxpool = nn.MaxPool2d(2, 2)
        self.Relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv1(x)
        x = self.Relu(x)
        x = self.conv2(x)
        x = self.Relu(x)
        x = self.dropout(x)

        return x


class UpConvolution(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(UpConvolution, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, (3, 3))
        self.conv2 = nn.Conv2d(output_channel, output_channel, (3, 3))
        self.convtranspose = nn.ConvTranspose2d(
            output_channel,
            output_channel // 2,
            (2, 2),
            (2, 2),
        )
        self.Relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Relu(x)
        x = self.conv2(x)
        x = self.Relu(x)
        x = self.dropout(x)
        x = self.convtranspose(x)

        return x


class LastConvolution(nn.Module):
    def __init__(self, input_channel, output_channel, num_classes):
        super(LastConvolution, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, (3, 3))
        self.conv2 = nn.Conv2d(output_channel, output_channel, (3, 3))
        self.conv1d = nn.Conv2d(output_channel, num_classes, (1, 1))
        self.Relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Relu(x)
        x = self.conv2(x)
        x = self.Relu(x)
        x = self.dropout(x)
        x = self.conv1d(x)

        return x


class UNet(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(UNet, self).__init__()
        self.simpleConv = SimpleConvolution(input_channel, 64)
        self.downConvBlock1 = DownConvolution(64, 128)
        self.downConvBlock2 = DownConvolution(128, 256)
        self.downConvBlock3 = DownConvolution(256, 512)
        self.midMaxpool = nn.MaxPool2d(2, 2)
        self.bridge = UpConvolution(512, 1024)
        self.upConvBlock1 = UpConvolution(1024, 512)
        self.upConvBlock2 = UpConvolution(512, 256)
        self.upConvBlock3 = UpConvolution(256, 128)
        self.lastConv = LastConvolution(128, 64, num_classes)

        self.n_channels = input_channel
        self.n_classes = num_classes

    def forward(self, x) -> Tensor:
        x_1 = self.simpleConv(x)
        x_2 = self.downConvBlock1(x_1)
        x_3 = self.downConvBlock2(x_2)
        x_4 = self.downConvBlock3(x_3)
        x_5 = self.midMaxpool(x_4)
        x_6 = self.bridge(x_5)
        crop_x_4 = crop_img(x_4, x_6)
        concat_x_4_6 = torch.cat((crop_x_4, x_6), 1)
        x_7 = self.upConvBlock1(concat_x_4_6)
        crop_x_3 = crop_img(x_3, x_7)
        concat_x_3_7 = torch.cat((crop_x_3, x_7), 1)
        x_8 = self.upConvBlock2(concat_x_3_7)
        crop_x_2 = crop_img(x_2, x_8)
        concat_x_2_8 = torch.cat((crop_x_2, x_8), 1)
        x_9 = self.upConvBlock3(concat_x_2_8)
        crop_x_1 = crop_img(x_1, x_9)
        concat_x_1_9 = torch.cat((crop_x_1, x_9), 1)
        out = self.lastConv(concat_x_1_9)

        return out


if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.no_grad()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    unet = UNet(1, 1).cuda()

    img = Image.open("../data/Montgomery/images/MCUCXR_0001_0.png")
    img_resized = img.resize((572, 572))

    convert_tensor = transforms.ToTensor()
    inp = convert_tensor(img_resized).to(device).unsqueeze(0)
    out = unet(inp)
    probs = torch.sigmoid(out)[0]
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img.size),
        transforms.ToTensor()
    ])

    full_mask = tf(probs.cpu()).squeeze()

    mask_np = (full_mask > 0.5).numpy()
    im_out = Image.fromarray(mask_np)
    im_out.save("test.png")


