import torch
import torch.nn.functional as F
from torch import nn
class FeatureFusionMHSA(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super(FeatureFusionMHSA, self).__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.head_dim = feature_dim // num_heads

        # 确保特征维度可以被头数整除
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"

        # 定义多头自注意力中的线性层
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        
        # 定义输出层
        self.fc_out = nn.Linear(feature_dim, feature_dim)

        # 定义层归一化
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, value, query):
        batch_size, feature_dim, seq_length = query.size()
        value = value.transpose(1, 2)
        query = query.transpose(1, 2)
        # 将输入进行归一化
        value_ln = self.layer_norm(value)
        query_ln = self.layer_norm(query)

        # 将输入分割成多头
        Q = self.query(query_ln).view(batch_size, seq_length, self.num_heads, self.head_dim)
        K = self.key(value_ln).view(batch_size, seq_length, self.num_heads, self.head_dim)
        V = self.value(value_ln).view(batch_size, seq_length, self.num_heads, self.head_dim)

        # 转置以得到适合点积的形状
        Q = Q.transpose(1, 2)  # (batch_size, num_heads, seq_length, head_dim)
        K = K.transpose(1, 2)  # (batch_size, num_heads, seq_length, head_dim)
        V = V.transpose(1, 2)  # (batch_size, num_heads, seq_length, head_dim)

        # 计算注意力得分
        attention = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention = torch.softmax(attention, dim=-1)

        # 通过注意力得分加权值向量
        out = torch.matmul(attention, V)  # (batch_size, num_heads, seq_length, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, feature_dim)  # (batch_size, seq_length, feature_dim)

        # 应用最后的线性层
        out = self.fc_out(out)
        out = out.transpose(1, 2)
        return out

# 初始化视频特征和文本特征
# video_features = torch.randn(batch_size, feature_dim, seq_length)
# text_features = torch.randn(batch_size, feature_dim, seq_length)

# # 初始化融合模型
# feature_fusion_model = FeatureFusionMHSA(feature_dim, num_heads)

# # 特征融合
# fused_features = feature_fusion_model(video_features, text_features)

# # 验证融合后的特征维度
# print(fused_features.shape)  # 输出融合后的特征维度以验证