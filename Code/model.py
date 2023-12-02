
import torch
from modules import *
import torch.nn.init as torch_init


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        # m.bias.data.fill_(0.1)

class XModel(nn.Module):
    def __init__(self, cfg):
        super(XModel, self).__init__()
        self.t = cfg.t_step
        self.self_attention = XEncoder(
            d_model=cfg.feat_dim,
            hid_dim=cfg.hid_dim,
            out_dim=cfg.out_dim,
            n_heads=cfg.head_num,
            win_size=cfg.win_size,
            dropout=cfg.dropout,
            gamma=cfg.gamma,
            bias=cfg.bias,
            norm=cfg.norm,
        )
        self.classifier = nn.Conv1d(cfg.out_dim, 1, self.t, padding=0)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / cfg.temp))
        self.apply(weight_init)

    def forward(self, x, seq_len):
        x_e, x_v = self.self_attention(x, seq_len)
        logits = F.pad(x_e, (self.t - 1, 0))
        logits = self.classifier(logits)

        logits = logits.permute(0, 2, 1)
        logits = torch.sigmoid(logits)

        return logits, x_v
    


# model test
if __name__ == '__main__':
    import torch
    import numpy as np
    # 模型配置
    class Config:
        def __init__(self):
            self.t_step = 10  # 替换为你的配置
            self.feat_dim = 1024  # 替换为你的配置
            self.hid_dim = 512  # 替换为你的配置
            self.out_dim = 256  # 替换为你的配置
            self.head_num = 4  # 替换为你的配置
            self.win_size = 3  # 替换为你的配置
            self.dropout = 0.1  # 替换为你的配置
            self.gamma = 2.0  # 替换为你的配置
            self.bias = True  # 替换为你的配置
            self.norm = 'layer'  # 替换为你的配置
            self.temp = 1.0  # 替换为你的配置

    # 创建模型实例
    cfg = Config()
    model = XModel(cfg)


    device = torch.device("cuda")
    model = model.to(device)
    # 生成随机数据
    batch_size = 1
    seq_len = 98
    input_shape = (batch_size, seq_len, cfg.feat_dim)
    x = torch.rand(input_shape).to(device)
    seq_len_tensor = torch.tensor([seq_len]).to(device)

    # 将模型置于评估模式
    model.eval()

    # 前向传播
    with torch.no_grad():
        logits, x_v = model(x, seq_len_tensor)

    # 打印输出的 logits 和 x_v 的形状
    print("Logits Shape:", logits.shape)
    print("x_v Shape:", x_v.shape)
