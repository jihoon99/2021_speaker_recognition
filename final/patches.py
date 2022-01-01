import torch.nn as nn
import torch
# from torchsummary import summary


# class Residual(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn

#     def forward(self, x):
#         return self.fn(x) + x

# def ConvMixer(dim, depth, kernel_size=9, patch_size=7, n_classes=1000):
#     return nn.Sequential(
#         nn.Conv2d(dim, dim, kernel_size=patch_size, stride=patch_size),
#         nn.GELU(),
#         nn.BatchNorm2d(dim),
#         *[nn.Sequential(
#                 Residual(nn.Sequential(
#                     nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
#                     nn.GELU(),
#                     nn.BatchNorm2d(dim)
#                 )),
#                 nn.Conv2d(dim, dim, kernel_size=1),
#                 nn.GELU(),
#                 nn.BatchNorm2d(dim)
#         ) for i in range(depth)],
#         print(2),
#         nn.AdaptiveAvgPool2d((1,1)),
#         print(3),
#         nn.Flatten(),
#         print(4),
#         nn.Linear(dim, n_classes),
#         print(5)
#     )
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixer(nn.Module):
    def __init__(self, dim, depth, kernel_size=9, patch_size=7, in_chans=1, num_classes=1, activation=nn.GELU, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = dim
        
        # self.head = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()


        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size),
            activation(),
            nn.BatchNorm2d(dim)
        )
        self.blocks = nn.Sequential(
            *[nn.Sequential(
                    Residual(nn.Sequential(
                        nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                        activation(),
                        nn.BatchNorm2d(dim)
                    )),
                    nn.Conv2d(dim, dim, kernel_size=1),
                    activation(),
                    nn.BatchNorm2d(dim)
            ) for i in range(depth)]
        )
        self.pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(dim* 2, dim),  # [128*2, 128]
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
          
    def forward_features(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pooling(x)
        return x
    def forward(self, x,y):
        x = self.forward_features(x)
        y = self.forward_features(y)
        t = torch.cat([x, y], dim=-1)

        return self.fc(t).squeeze(-1) # 128*2
        # x = self.head(x)

        return x

# if __name__ == "__main__":
#     model = ConvMixer(128, 34)
#     summary(model, [(1,224,224), (1,224,224)])