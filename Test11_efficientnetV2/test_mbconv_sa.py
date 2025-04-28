import torch
import torch.nn as nn
from model import MBConv, SpatialAttention, ConvBNAct  # 假设你的模型代码保存在 model.py 文件中


def test_mbconv():
    # 设置一些模型参数
    kernel_size = 3
    input_c = 32  # 输入通道数
    out_c = 64  # 输出通道数
    expand_ratio = 4  # 扩展倍数
    stride = 1  # 步幅
    se_ratio = 0.25  # SE模块比例
    drop_rate = 0.2  # Drop路径概率
    norm_layer = nn.BatchNorm2d  # 使用批归一化

    # 创建MBConv模型
    model = MBConv(
        kernel_size=kernel_size,
        input_c=input_c,
        out_c=out_c,
        expand_ratio=expand_ratio,
        stride=stride,
        se_ratio=se_ratio,
        drop_rate=drop_rate,
        norm_layer=norm_layer
    )

    # 打印模型结构
    print(model)

    # 生成一个假输入，尺寸为 (batch_size, channels, height, width)，比如 (8, 32, 224, 224)
    x = torch.randn(8, input_c, 224, 224)  # 批次大小为 8，输入通道数为 32，图片尺寸为 224x224

    # 将输入传入模型
    output = model(x)

    # 打印输出的形状，检查是否符合预期
    print(f'Output shape: {output.shape}')


if __name__ == "__main__":
    test_mbconv()
