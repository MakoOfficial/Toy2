import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

def get_ResNet():
    """获得主干网络ResNet50"""
    model = resnet50(pretrained=True)
    output_channels = model.fc.in_features
    model = list(model.children())[:-2]
    return model, output_channels

class My_attention(nn.Module):
    """自定义核大小卷积核"""
    def __init__(self, input_channels, kernel_size=1) -> None:
        super().__init__()
        self.my_attention = nn.Sequential(
            nn.Conv2d(input_channels, 1, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU()
        )

    def forward(self, x):
        return self.my_attention(x)


class MMCA_module(nn.Module):
    """构建MMCA模块，MMCA包括DR和MRA，DR需要降维因子reduction[]，还有DR的层数level，故需要参数：输入通道，降维因子reduction，层数level
        注：MMCA并不改编通道数"""
    def __init__(self, input_channels, reduction=[16], level=1) -> None:
        super().__init__()
        modules = []
        for i in range(level):
            output_channels = input_channels // reduction[i]
            modules.append(nn.Conv2d(input_channels, output_channels, kernel_size = 1))
            modules.append(nn.BatchNorm2d(output_channels))
            modules.append(nn.ReLU())
            input_channels = output_channels

        self.DR = nn.Sequential(*modules)
        self.MRA1 = My_attention(input_channels, 1)
        self.MRA3 = My_attention(input_channels, 3)
        self.MRA5 = My_attention(input_channels, 5)
        self.last_conv = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=1),
            nn.Sigmoid()  
        )

    def forward(self, x):
        input = x.clone()
        x = self.DR(x)
        x = torch.cat([self.MRA1(x), self.MRA3(x), self.MRA5(x)], dim=1)

        x = self.last_conv(x)
        #   F*(1-A) = F - F*A
        return (1 - x), (input - input * x)
        # return (input - input * x)
        
class DW_conv(nn.Module):
    def __init__(self, nin, nout, stride=2, kernel_size=3, padding=1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels=nin, out_channels=nin, kernel_size=kernel_size, stride=stride, padding=padding, groups=nin, )
        self.pointwise = nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


# class KMEANS:
#     def __init__(self, n_clusters=20, max_iter=None, verbose=True, device = torch.device("cpu")):
#         """the implement of K-Meas"""
#         self.n_cluster = n_clusters
#         self.n_clusters = n_clusters
#         self.labels = None
#         self.dists = None  # shape: [x.shape[0],n_cluster]
#         self.centers = None
#         self.variation = torch.Tensor([float("Inf")]).to(device)
#         self.verbose = verbose
#         self.started = False
#         self.representative_samples = None
#         self.max_iter = max_iter
#         self.count = 0
#         self.device = device

#     def fit(self, x):
#         print("K-Means start!!!")
#         # 随机选择初始中心点，想更快的收敛速度可以借鉴sklearn中的kmeans++初始化方法
#         bacth_size = x.shape[0]
#         self.count = 0
#         x = torch.reshape(x, [bacth_size, -1])
#         if self.centers is None:
#             init_row = torch.randint(0, bacth_size, (self.n_clusters,)).to(self.device)
#             init_points = x[init_row]
#             self.centers = init_points
#         while True:
#             # 聚类标记
#             self.nearest_center(x)
#             # 更新中心点
#             self.update_center(x)
#             if self.verbose:
#                 print(self.variation, torch.argmin(self.dists, dim=0))
#             if torch.abs(self.variation) < 1e-3 and self.max_iter is None:
#                 break
#             elif self.max_iter is not None and self.count == self.max_iter:
#                 break

#             self.count += 1
#         print("K-Means is over!!!")
#         self.representative_sample()

#     def nearest_center(self, x):
#         labels = torch.empty((x.shape[0],)).long().to(self.device)
#         dists = torch.empty((0, self.n_clusters)).to(self.device)
#         for i, sample in enumerate(x):
#             # print(f"centers is {self.centers}")
#             dist = torch.sum(torch.mul(sample - self.centers, sample - self.centers), dim=1)
#             labels[i] = torch.argmin(dist)
#             dists = torch.cat([dists, dist.unsqueeze(0)], dim=0)
#         self.labels = labels
#         if self.started:
#             self.variation = torch.sum(self.dists - dists)
#         self.dists = dists
#         self.started = True

#     def update_center(self, x):
#         centers = torch.empty((0, x.shape[1])).to(self.device)
#         for i in range(self.n_clusters):
#             mask = self.labels == i
#             cluster_samples = x[mask]
#             if min(cluster_samples.shape) != 0:
#                 centers = torch.cat([centers, torch.mean(cluster_samples, dim=0).unsqueeze(0)], dim=0)
#             else:
#                 centers = torch.cat([centers, self.centers[i].unsqueeze(0)], dim=0)
#         self.centers = centers

#     def representative_sample(self):
#         # 查找距离中心点最近的样本，作为聚类的代表样本，更加直观
#         self.representative_samples = torch.argmin(self.dists, (0))

#     def fine_tune(self):
#         self.centers = self.centers.to(torch.device("cuda"))

class Multi_Sacle_Fusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 256
        self.fusion1_upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.fusion1_upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear"),
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.fusion1_upsample3 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode="bilinear"),
            nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        # 512
        self.fusion2_DW = nn.Sequential(
            DW_conv(256, 512, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.fusion2_upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.fusion2_upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear"),
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        # 1024
        self.fusion3_DW1 = nn.Sequential(
            DW_conv(256, 1024, stride=4, kernel_size=5, padding=2),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        self.fusion3_DW2 = nn.Sequential(
            DW_conv(512, 1024, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        self.fusion3_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        # 2048
        self.fusion4_DW1 = nn.Sequential(
            DW_conv(256, 2048, stride=8, kernel_size=9, padding=4),
            nn.BatchNorm2d(2048),
            nn.ReLU()
            )
        self.fusion4_DW2 = nn.Sequential(
            DW_conv(512, 2048, stride=4, kernel_size=5, padding=2),
            nn.BatchNorm2d(2048),
            nn.ReLU()
            )
        self.fusion4_DW3 = nn.Sequential(
            DW_conv(1024, 2048, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU()
            )
    
    def forward(self, feature1, feature2, feature3, feature4):
        fusion_feature1 = feature1 + self.fusion1_upsample1(feature2)
        fusion_feature1 = fusion_feature1 + self.fusion1_upsample2(feature3)
        fusion_feature1 = fusion_feature1 + self.fusion1_upsample3(feature4)

        fusion_feature2 = self.fusion2_DW(feature1) + feature2
        fusion_feature2 = fusion_feature2 + self.fusion2_upsample1(feature3)
        fusion_feature2 = fusion_feature2 + self.fusion2_upsample2(feature4)

        fusion_feature3 = self.fusion3_DW1(feature1) + self.fusion3_DW2(feature2)
        fusion_feature3 = fusion_feature3 + feature3
        fusion_feature3 = fusion_feature3 + self.fusion3_upsample(feature4)

        fusion_feature4 = self.fusion4_DW1(feature1) + self.fusion4_DW2(feature2)
        fusion_feature4 = fusion_feature4 + self.fusion4_DW3(feature3)
        fusion_feature4 = fusion_feature4 + feature4

        return fusion_feature1, fusion_feature2, fusion_feature3, fusion_feature4
    

class Toy(nn.Module):
    """主模型MMANet的在输入到GA前的部分"""
    def __init__(self, genderSize, backbone, out_channels) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.backbone1 = nn.Sequential(*backbone[0:5])
        self.MMCA1 = MMCA_module(256)
        self.backbone2 = backbone[5]
        self.MMCA2 = MMCA_module(512, reduction=[4, 8], level=2)
        self.backbone3 = backbone[6]
        self.MMCA3 = MMCA_module(1024, reduction=[8, 8], level=2)
        self.backbone4 = backbone[7]
        self.MMCA4 = MMCA_module(2048, reduction=[8, 16], level=3)

        self.MSA = Multi_Sacle_Fusion()

        self.gender_encoder = nn.Linear(1, genderSize)
        self.gender_BN = nn.BatchNorm1d(genderSize)
        self.MLP1 = nn.Sequential(
            nn.Linear(256 + genderSize, 228),
            nn.Softmax()
        )
        self.MLP2 = nn.Sequential(
            nn.Linear(512 + genderSize, 228),
            nn.Softmax()
        )
        self.MLP3 = nn.Sequential(
            nn.Linear(1024 + genderSize, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 228),
            nn.Softmax()
        )
        self.MLP4 = nn.Sequential(
            nn.Linear(2048 + genderSize, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 228),
            nn.Softmax()
        )

    def Residual_Representation(self, feature_map, prototype):
        shape = feature_map.shape
        x = torch.reshape(feature_map, [feature_map.shape[0], -1])
        labels = torch.empty((x.shape[0],)).long().to(feature_map.device)

        for i, sample in enumerate(x):
            # print(f"centers is {self.centers}")
            dist = torch.sum(torch.mul(sample - prototype, sample - prototype), dim=1)
            labels[i] = torch.argmin(dist)
        
        return torch.reshape((x - prototype[labels]), shape=shape)

    def forward(self, image, gender):
        AM1, F1 = self.MMCA1(self.backbone1(image))
        AM2, F2 = self.MMCA2(self.backbone2(F1))
        AM3, F3 = self.MMCA3(self.backbone3(F2))
        AM4, F4 = self.MMCA4(self.backbone4(F3))
        
        # feature_map_1 = self.backbone1(image)
        # res_repre_1 = self.Residual_Representation(feature_map_1, self.K_means1.centers)
        # print(feature_map_1.shape)
        # feature_map_2 = self.backbone2(feature_map_1)
        # res_repre_2 = self.Residual_Representation(feature_map_2, self.K_means2.centers)
        # print(feature_map_2.shape)
        # feature_map_3 = self.backbone3(feature_map_2)
        # res_repre_3 = self.Residual_Representation(feature_map_3, self.K_means3.centers)
        # print(feature_map_3.shape)
        # feature_map_4 = self.backbone4(feature_map_3)
        # res_repre_4 = self.Residual_Representation(feature_map_4, self.K_means4.centers)
        # print(feature_map_4.shape)
        fusion_feature1, fusion_feature2, fusion_feature3, fusion_feature4 = self.MSA(F1, F2, F3, F4)
        # fusion_res1, fusion_res2, fusion_res3, fusion_res4 = self.MSA(res_repre_1, res_repre_2, res_repre_3, res_repre_4)

        gender_encode = self.gender_encoder(gender)
        gender_encode = self.gender_BN(gender_encode)
        gender_encode = F.relu(gender_encode)

        # feature_for_attention1 = torch.cat([fusion_feature1, fusion_res1], dim=1)
        feature_for_reg1 = torch.cat([torch.squeeze(F.adaptive_avg_pool2d(fusion_feature1, 1)), gender_encode], dim=1)
        # feature_for_attention2 = torch.cat([fusion_feature2, fusion_res2], dim=1)
        feature_for_reg2 = torch.cat([torch.squeeze(F.adaptive_avg_pool2d(self.MMCA2(fusion_feature2), 1)), gender_encode], dim=1)
        # feature_for_attention3 = torch.cat([fusion_feature3, fusion_res3], dim=1)
        feature_for_reg3 = torch.cat([torch.squeeze(F.adaptive_avg_pool2d(self.MMCA3(fusion_feature3), 1)), gender_encode], dim=1)
        # feature_for_attention4 = torch.cat([fusion_feature4, fusion_res4], dim=1)
        feature_for_reg4 = torch.cat([torch.squeeze(F.adaptive_avg_pool2d(self.MMCA4(fusion_feature4), 1)), gender_encode], dim=1)

        return self.MLP1(feature_for_reg1), self.MLP2(feature_for_reg2), self.MLP3(feature_for_reg3), self.MLP4(feature_for_reg4)
        
        # output_beforeGA = F.softmax(output_beforeGA)
        # distribute = torch.arange(0, 240)
        # output_beforeGA = (output_beforeGA*distribute).sum(dim=1)

        # return AM1, AM2, AM3, AM4, feature_map, texture, gender_encode, output_beforeGA
        # return AM1, AM2, AM3, AM4, output_beforeGA
        # return output_beforeGA
    # 加入微调函数
    def fine_tune(self, need_fine_tune = True):
        self.train(need_fine_tune)
        # self.K_means1.fine_tune()
        # self.K_means2.fine_tune()
        # self.K_means3.fine_tune()
        # self.K_means4.fine_tune()



if __name__ == '__main__':
    DW_conv(256, 2048, stride=8, kernel_size=9, padding=4)
