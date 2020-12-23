import torch.nn as nn


class PredictionModel(nn.Module):
    def __init__(self, channel_list=[32, 32, 64, 64, 128, 128, 256, 256]):
        super().__init__()
        num_features = 28
        in_channels = 3
        conv_layers = []

        # Use this method to build the network layers
        def conv_block(c, h): return [nn.BatchNorm2d(h), nn.Conv2d(h, c, 5, 2, 2), nn.ReLU(True)]

        # Build the network
        for out_channel in channel_list:
            conv_layers += conv_block(out_channel, in_channels)
            in_channels = out_channel
        self.classifier = nn.Sequential(*conv_layers, nn.Conv2d(in_channels, num_features, 1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, img):
        z = self.classifier(img)
        z = z.mean(dim=[2, 3])
        return self.softmax(z)


if __name__ == "__main__":
    import torch

    # Check that the network works
    mymodel = PredictionModel()
    a = torch.randn(8, 3, 200, 200)
    z = mymodel(a)
    print(a.shape)
    print(z.shape)
    print(z)
