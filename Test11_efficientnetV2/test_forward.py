import torch
from torch.utils.tensorboard import SummaryWriter
from model import EfficientNetV2  # 确保路径正确


def main():

    model_config = [[4, 3, 1, 1, 32, 32, 0, 0],
                    [7, 3, 2, 4, 32, 64, 0, 0],
                    [7, 3, 2, 4, 64, 96, 0, 0],
                    [10, 3, 2, 4, 96, 192, 1, 0.25],
                    [19, 3, 1, 6, 192, 224, 1, 0.25],
                    [25, 3, 2, 6, 224, 384, 1, 0.25],
                    [7, 3, 1, 6, 384, 640, 1, 0.25]]
    # 创建模型
    model = EfficientNetV2(
        model_cnf=model_config,
        num_classes=3,
        mid_layer_idx=10
    )

    model.eval()  # 设置为评估模式

    # 随机输入数据，模拟 batch_size=1 的 RGB 图像，大小224x224（可以根据实际修改）
    dummy_input = torch.randn(1, 3, 224, 224)

    # 初始化 TensorBoard writer
    writer = SummaryWriter(log_dir='./runs/efficientnetv2_test')

    # 添加模型结构图
    writer.add_graph(model, dummy_input)

    print("模型计算图已写入 TensorBoard logs.")
    print("你可以通过以下命令启动 TensorBoard：")
    print("tensorboard --logdir=./runs")

    writer.close()


if __name__ == '__main__':
    main()
