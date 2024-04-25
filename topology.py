import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment


# 定义TopoSeg模型类
class TopoSeg(nn.Module):
    def __init__(self, num_classes):
        super(TopoSeg, self).__init__()
        # 语义分支
        self.semantic_branch = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        # 三类分支
        self.three_class_branch = nn.Sequential(
            # 与semantic_branch结构相同
        )

        # 语义分支输出层
        self.semantic_out = nn.Conv2d(64, num_classes, kernel_size=1)
        # 三类分支输出层
        self.three_class_out = nn.Conv2d(64, 3, kernel_size=1)
        # 拓扑感知模块TAM
        self.tam = TopologyAwareModule()
        # 自适应拓扑感知选择ATS
        self.ats = AdaptiveTopologySelection()

    def forward(self, x):
        # 语义分支前向传播
        semantic_feat = self.semantic_branch(x)
        semantic_out = self.semantic_out(semantic_feat)
        # 三类分支前向传播
        three_class_feat = self.three_class_branch(x)
        three_class_out = self.three_class_out(three_class_feat)
        three_class_prob = F.softmax(three_class_out, dim=1)
        # ATS筛选
        selected_regions = self.ats(three_class_prob)
        # TAM处理
        topo_loss = self.tam(three_class_prob, selected_regions)
        return semantic_out, three_class_out, topo_loss


# 定义拓扑感知模块TAM类
class TopologyAwareModule(nn.Module):
    def __init__(self):
        super(TopologyAwareModule, self).__init__()
        # 实现拓扑编码和持续同调
        self.betti_numbers = None
        self.thresholds = torch.linspace(0, 1, steps=100)

    def forward(self, three_class_prob, selected_regions):
        # 对选定区域进行拓扑编码
        topo_codes = self.topo_encoding(three_class_prob, selected_regions)
        # 计算持续同调
        barcodes = self.persistent_homology(topo_codes)
        # 构建拓扑感知损失
        topo_loss = self.topo_loss(barcodes)
        return topo_loss

    def topo_encoding(self, three_class_prob, selected_regions):
        # 将三类概率图分解为内部、边界、并集三个通道
        inside_prob = three_class_prob[:, 0, :, :][selected_regions]
        boundary_prob = three_class_prob[:, 1, :, :][selected_regions]
        union_prob = three_class_prob[:, 0:1, :, :][selected_regions]
        # 计算每个通道的贝蒂数
        inside_betti = self.compute_betti_numbers(inside_prob)
        boundary_betti = self.compute_betti_numbers(boundary_prob)
        union_betti = self.compute_betti_numbers(union_prob)
        # 拼接贝蒂数向量
        topo_codes = torch.cat([inside_betti, boundary_betti, union_betti], dim=-1)
        return topo_codes

    def compute_betti_numbers(self, prob):
        # 阈值化概率图
        binarized = (prob > self.thresholds.to(prob.device).view(-1, 1, 1, 1)).float()
        # 统计连通分量和空洞数量
        b0 = self.count_components(binarized)
        b1 = self.count_holes(binarized)
        betti_numbers = torch.stack([b0, b1], dim=-1)
        self.betti_numbers = betti_numbers
        return betti_numbers

    @staticmethod
    def count_components(binarized):
        # 定义连通分量标记函数
        def dfs(img, visited, i, j):
            rows, cols = img.shape
            if i < 0 or i >= rows or j < 0 or j >= cols or visited[i][j] or img[i][j] == 0:
                return
            visited[i][j] = True
            dfs(img, visited, i + 1, j)
            dfs(img, visited, i - 1, j)
            dfs(img, visited, i, j + 1)
            dfs(img, visited, i, j - 1)

        # 统计连通分量数量
        num_components = 0
        rows, cols = binarized.shape
        visited = [[False] * cols for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                if not visited[i][j] and binarized[i][j] == 1:
                    dfs(binarized, visited, i, j)
                    num_components += 1
        return num_components

    def count_holes(self, binarized):
        # 计算欧拉数
        euler_number = self.compute_euler_number(binarized)
        # 空洞数量 = 连通分量数量 - 欧拉数
        holes = self.count_components(binarized) - euler_number
        return holes

    @staticmethod
    def compute_euler_number(binarized):
        # 计算像素点数量
        num_pixels = torch.sum(binarized, dim=[1, 2, 3])
        # 计算边数量
        num_edges = torch.sum(binarized[:, :, :-1, :] * binarized[:, :, 1:, :], dim=[1, 2, 3]) + \
                    torch.sum(binarized[:, :, :, :-1] * binarized[:, :, :, 1:], dim=[1, 2, 3])
        # 欧拉数 = 像素点数量 - 边数量
        euler_number = num_pixels - num_edges
        return euler_number

    def persistent_homology(self, topo_codes):
        # 从拓扑编码计算持续同调条形码
        barcodes = []
        for i in range(topo_codes.shape[1]):
            code = topo_codes[:, i, :]
            barcode = self.compute_barcode(code)
            barcodes.append(barcode)
        barcodes = torch.stack(barcodes, dim=0)
        return barcodes

    def compute_barcode(self, code):
        # 计算单个拓扑编码的条形码
        birth = []
        death = []
        for i in range(code.shape[0] - 1):
            if code[i] < code[i + 1]:
                birth.append(self.thresholds[i])
            elif code[i] > code[i + 1]:
                death.append(self.thresholds[i])
        birth = torch.tensor(birth)
        death = torch.tensor(death)
        barcode = torch.stack([birth, death], dim=-1)
        return barcode

    def topo_loss(self, barcodes):
        # 基于条形码计算拓扑损失
        gt_barcodes = self.gt_barcodes.to(barcodes.device)
        # 匹配预测条形码与GT条形码
        matched_barcodes, matched_gt_barcodes = self.match_barcodes(barcodes, gt_barcodes)
        # 计算匹配条形码的长度差异
        matched_diff = torch.sum(torch.abs(matched_barcodes - matched_gt_barcodes), dim=-1)
        # 计算未匹配条形码的长度
        unmatched_len = torch.sum(barcodes[..., 1] - barcodes[..., 0], dim=-1)
        topo_loss = torch.mean(matched_diff) + torch.mean(unmatched_len)
        return topo_loss

    @staticmethod
    def match_barcodes(barcodes, gt_barcodes):
        # 计算条形码之间的距离矩阵
        distances = torch.cdist(barcodes, gt_barcodes)
        # 将距离矩阵转为numpy数组
        distances_np = distances.cpu().numpy()
        # 使用匈牙利算法计算最优匹配
        row_ind, col_ind = linear_sum_assignment(distances_np)
        # 获取匹配的条形码
        matched_barcodes = barcodes[row_ind]
        matched_gt_barcodes = gt_barcodes[col_ind]
        return matched_barcodes, matched_gt_barcodes


# 定义自适应拓扑感知选择ATS类
class AdaptiveTopologySelection(nn.Module):
    def __init__(self):
        super(AdaptiveTopologySelection, self).__init__()
        # 实现图像级自适应选择
        self.img_threshold = None
        # 实现区域级自适应选择
        self.region_size = (32, 32)
        self.region_threshold = None

    def forward(self, three_class_prob):
        # 图像级选择
        selected_images = self.image_selection(three_class_prob)
        # 区域级选择
        selected_regions = self.region_selection(three_class_prob, selected_images)
        return selected_regions

    def image_selection(self, three_class_prob):
        # 计算每张图像的拓扑误差
        topo_errors = self.compute_topo_errors(three_class_prob)
        # 根据自适应阈值筛选拓扑误差大的图像
        self.img_threshold = self.adapt_threshold(topo_errors)
        selected_images = topo_errors > self.img_threshold
        return selected_images

    def compute_topo_errors(self, three_class_prob):
        # 计算三类概率图与GT的贝蒂数之差
        inside_prob = three_class_prob[:, 0, :, :]
        boundary_prob = three_class_prob[:, 1, :, :]
        union_prob = three_class_prob[:, 0:1, :, :]
        inside_betti = self.compute_betti_numbers(inside_prob)
        boundary_betti = self.compute_betti_numbers(boundary_prob)
        union_betti = self.compute_betti_numbers(union_prob)
        gt_inside_betti = self.gt_betti_numbers[:, 0:1]
        gt_boundary_betti = self.gt_betti_numbers[:, 1:2]
        gt_union_betti = self.gt_betti_numbers[:, 2:3]
        inside_error = torch.sum(torch.abs(inside_betti - gt_inside_betti), dim=-1)
        boundary_error = torch.sum(torch.abs(boundary_betti - gt_boundary_betti), dim=-1)
        union_error = torch.sum(torch.abs(union_betti - gt_union_betti), dim=-1)
        topo_errors = inside_error + boundary_error + union_error
        return topo_errors

    @staticmethod
    def adapt_threshold(topo_errors, ratio=0.25):
        # 计算拓扑误差的平均值和标准差
        mean_error = torch.mean(topo_errors)
        std_error = torch.std(topo_errors)
        # 自适应阈值 = 平均值 + ratio * 标准差
        threshold = mean_error + ratio * std_error
        return threshold

    def region_selection(self, three_class_prob, selected_images):
        # 将图像划分为多个区域
        regions = self.split_regions(three_class_prob)
        # 计算每个区域的拓扑误差
        region_errors = self.compute_region_errors(regions)
        # 根据自适应阈值筛选拓扑误差大的区域
        self.region_threshold = self.adapt_threshold(region_errors)
        selected_regions = region_errors > self.region_threshold
        # 将区域选择结果映射回图像维度
        selected_regions = self.map_regions(selected_regions, selected_images)
        return selected_regions

    def split_regions(self, three_class_prob):
        batch_size, _, height, width = three_class_prob.shape
        # 计算区域的行数和列数
        rows = height // self.region_size[0]
        cols = width // self.region_size[1]
        # 使用unfold操作划分区域
        regions = three_class_prob.unfold(2, self.region_size[0], self.region_size[0]).unfold(3, self.region_size[1],
                                                                                              self.region_size[1])
        regions = regions.contiguous().view(batch_size, 3, rows, cols, self.region_size[0], self.region_size[1])
        return regions

    def compute_region_errors(self, regions):
        batch_size, _, rows, cols, _, _ = regions.shape
        # 计算每个区域的贝蒂数
        inside_betti = self.compute_betti_numbers(
            regions[:, 0, ...].view(batch_size * rows * cols, 1, self.region_size[0], self.region_size[1]))
        boundary_betti = self.compute_betti_numbers(
            regions[:, 1, ...].view(batch_size * rows * cols, 1, self.region_size[0], self.region_size[1]))
        union_betti = self.compute_betti_numbers(
            regions[:, 0:1, ...].view(batch_size * rows * cols, 1, self.region_size[0], self.region_size[1]))
        # 重塑贝蒂数的形状
        inside_betti = inside_betti.view(batch_size, rows, cols, -1)
        boundary_betti = boundary_betti.view(batch_size, rows, cols, -1)
        union_betti = union_betti.view(batch_size, rows, cols, -1)
        # 计算GT贝蒂数
        gt_inside_betti = self.gt_betti_numbers[:, 0:1]
        gt_boundary_betti = self.gt_betti_numbers[:, 1:2]
        gt_union_betti = self.gt_betti_numbers[:, 2:3]
        # 计算贝蒂数之差
        inside_error = torch.sum(torch.abs(inside_betti - gt_inside_betti), dim=-1)
        boundary_error = torch.sum(torch.abs(boundary_betti - gt_boundary_betti), dim=-1)
        union_error = torch.sum(torch.abs(union_betti - gt_union_betti), dim=-1)
        region_errors = inside_error + boundary_error + union_error
        return region_errors

    def map_regions(self, selected_regions, selected_images):
        batch_size, rows, cols = selected_regions.shape
        # 将区域选择结果转为布尔掩码
        region_masks = selected_regions.view(batch_size, 1, rows, cols).repeat(1, 3, 1, 1)
        region_masks = F.interpolate(region_masks.float(), size=(self.image_size[0], self.image_size[1]),
                                     mode='nearest').bool()
        # 将区域掩码应用到选择的图像上
        selected_masks = torch.zeros_like(selected_images, dtype=torch.bool)
        selected_masks[selected_images] = region_masks[selected_images]
        return selected_masks
