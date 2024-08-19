import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

# Transformerモデルの自作
class AddPositionalEncoding(nn.Module):
    def __init__(self, d_model, time_step, device = torch.device("cpu")):
        super().__init__()
        self.d_model = d_model
        self.time_step = time_step
        positional_encoding_weight = self._initialize_weight().to(device)
        self.register_buffer("positional_encoding_weight", positional_encoding_weight)

    def forward(self, x):
        seq_len = x.size(1)
        tmp = self.positional_encoding_weight[:seq_len, :].unsqueeze(0)
        return x + tmp

    def _get_positional_encoding(self, pos, i):
        w = pos / (10000 ** (((2 * i) // 2) / self.d_model))
        if i % 2 == 0:
            return np.sin(w)
        else:
            return np.cos(w)

    def _initialize_weight(self):
        positional_encoding_weight = [
            [self._get_positional_encoding(pos, i) for i in range(1, self.d_model + 1)]
            for pos in range(1, self.time_step + 1)
        ]
        return torch.tensor(positional_encoding_weight).float()


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout_rate=0.1, device=torch.device('cpu')):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff, bias=True, device=device)
        self.W2 = nn.Linear(d_ff, d_model, bias=True, device=device)
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, train_flg=True):
        output = self.W2(F.relu(self.W1(x)))
        
        if (train_flg):
            output = self.dropout(output)
        else:
            output = output * (1 - self.dropout_rate)
            
        return output


class ScaledDotProductAttention(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super().__init__()
        self.device = device

    def forward(self, Q, K, V, mask):
        d_k = K.shape[3]                                    # d_model
        
        K_t = torch.transpose(K, 3, 2)                      # [batch_size, heads_num, d_k, time_step]
        QK = ((Q @ K_t) / math.sqrt(d_k))                   # [batch_size, heads_num, time_step, time_step]
        QK[:, :, mask] = -999999999.0
        attention_score = F.softmax(QK, dim=3)              # [batch_size, heads_num, time_step, time_step]
        
        output = attention_score @ V                        # [batch_size, heads_num, time_step, d_k]

        return output, attention_score


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, heads_num=8, dropout_rate=0.1, device=torch.device('cpu')):
        super().__init__()
        self.heads_num = heads_num
        self.d_k = int(d_model / heads_num)
        self.W_Q = nn.Linear(d_model, d_model, bias=False, device=device)
        self.W_K = nn.Linear(d_model, d_model, bias=False, device=device)
        self.W_V = nn.Linear(d_model, d_model, bias=False, device=device)
        self.W_O = nn.Linear(heads_num*self.d_k, d_model, bias=False, device=device)
        self.attention = ScaledDotProductAttention(device)
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, Q, K, V, mask, train_flg=True):
        B, T, D = Q.shape
        
        Q = self.W_Q(Q)                                                   # [batch_size, time_step, d_model]
        K = self.W_K(K)                                                   # [batch_size, time_step, d_model]
        V = self.W_V(V)                                                   # [batch_size, time_step, d_model]

        QW = Q.view(B, T, self.heads_num, self.d_k).transpose(1,2)        # [batch_size, heads_num, time_step, d_k]
        KW = K.view(B, T, self.heads_num, self.d_k).transpose(1,2)        # [batch_size, heads_num, time_step, d_k]
        VW = V.view(B, T, self.heads_num, self.d_k).transpose(1,2)        # [batch_size, heads_num, time_step, d_k]

        head_i, attention_score = self.attention(QW, KW, VW, mask)        # [batch_size, heads_num, time_step, d_k]
        head_cat = head_i.permute(0, 2, 1, 3).contiguous().view(head_i.size(0), head_i.size(2), -1) # [batch_size, time_step, heads_num*d_k]
        output = self.W_O(head_cat)                                       # [batch_size, time_step, d_model]
        if (train_flg):
            output = self.dropout(output)
        else:
            output = output * (1 - self.dropout_rate)

        return output, attention_score


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, heads_num=8, dropout_rate=0.1, device=torch.device('cpu')):
        super().__init__()
        self.multiheadattention = MultiHeadAttention(d_model, heads_num, dropout_rate, device)
        self.norm1 = nn.LayerNorm(d_model, device=device)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout_rate, device)
        self.norm2 = nn.LayerNorm(d_model, device=device)
    
    def forward(self, x, mask, train_flg=True):
        _x = x
        x, attention = self.multiheadattention(x, x, x, mask, train_flg)
        x = self.norm1(x + _x)

        _x = x
        x = self.ffn(x, train_flg)
        x = self.norm1(x + _x)

        return x, attention


class Encoder(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, heads_num=8, layers_num=6, dropout_rate=0.1, device=torch.device('cpu')):
        super().__init__()
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, d_ff, heads_num, dropout_rate, device) for _ in range(layers_num)])

    def forward(self, x, mask, train_flg=True):
        attention_record = []
        for encoder_layer in self.encoder_layers:
            x, attention = encoder_layer(x, mask, train_flg)
            attention_record.append(attention)

        return x, attention_record


# 並列+ST
class myModel1(nn.Module):
    def __init__(self, input_dim=4, output_dim=4, emb_dim=64, time_step=10, d_model=512, d_ff=2048, heads_num=8, layers_num=6, dropout_rate=0.1, device=torch.device('cpu')):
        super().__init__()
        self.emb_dim = emb_dim
        self.d_model = d_model
        self.time_step = time_step
        self.linear1 = nn.Linear(input_dim, emb_dim, bias=True, device=device)
        self.linear2 = nn.Linear(emb_dim, d_model, bias=True, device=device)
        self.spatial_transformer_1 = Encoder(d_model, d_ff, heads_num, layers_num, dropout_rate, device)
        self.pos_encoding = AddPositionalEncoding(d_model, time_step, device=device)
        self.temporal_transformer_1 = Encoder(d_model, d_ff, heads_num, layers_num, dropout_rate, device)
        self.linear3 = nn.Linear(d_model*2, d_model, bias=False, device=device)
        self.spatial_transformer_2 = Encoder(d_model, d_ff, heads_num, layers_num, dropout_rate, device)
        self.temporal_transformer_2 = Encoder(d_model, d_ff, heads_num, layers_num, dropout_rate, device)
        self.linear_dec1 = nn.Linear(d_model, emb_dim, bias=True, device=device)
        self.linear_dec2 = nn.Linear(emb_dim, output_dim, bias=True, device=device)
        self.device = device

    def forward(self, x, s_mask, t_mask, train_flg=True):
        agents, T = x.shape[0], x.shape[1]

        x = self.linear2(self.linear1(x))                                                                      # [agents, time_step, d_model]

        spatial_x, s_attention1 = self.spatial_transformer_1(x.transpose(0,1), s_mask, train_flg)              # [time_step, agents, d_model]
        temporal_x, t_attention1 = self.temporal_transformer_1(self.pos_encoding(x), t_mask, train_flg)        # [agents, time_step, d_model]
        x = torch.cat([spatial_x.transpose(0,1), temporal_x], dim=2)                                           # [agents, time_step, d_model*2]
        x = self.linear3(x)                                                                                    # [agents, time_step, d_model]

        x, s_attention2 = self.spatial_transformer_2(x.transpose(0,1), s_mask, train_flg)                      # [time_step, agents, d_model]
        x, t_attention2 = self.temporal_transformer_2(self.pos_encoding(x.transpose(0,1)), t_mask, train_flg)  # [agents, time_step, d_model]

        if train_flg:
            x = x + torch.Tensor(np.random.normal(loc=0.0, scale=1.0, size=(agents, T, self.d_model))).to(self.device)

        output = self.linear_dec2(self.linear_dec1(x))                                                         # [agents, time_step, output_dim]
        output[:,:,[1,3]] = torch.exp(output[:,:,[1,3]])                                                 # 標準偏差のみ指数関数を通すことで負になるのを防ぐ

        return output, [t_attention1, t_attention2, s_attention1, s_attention2]  # [N, T, 4] 観察時間先T時刻のN人の(μx, σx, μy, σy)


# 並列+TS
class myModel2(nn.Module):
    def __init__(self, input_dim=4, output_dim=4, emb_dim=64, time_step=10, d_model=512, d_ff=2048, heads_num=8, layers_num=6, dropout_rate=0.1, device=torch.device('cpu')):
        super().__init__()
        self.emb_dim = emb_dim
        self.d_model = d_model
        self.time_step = time_step
        self.linear1 = nn.Linear(input_dim, emb_dim, bias=True, device=device)
        self.linear2 = nn.Linear(emb_dim, d_model, bias=True, device=device)
        self.spatial_transformer_1 = Encoder(d_model, d_ff, heads_num, layers_num, dropout_rate, device)
        self.pos_encoding = AddPositionalEncoding(d_model, time_step, device=device)
        self.temporal_transformer_1 = Encoder(d_model, d_ff, heads_num, layers_num, dropout_rate, device)
        self.linear3 = nn.Linear(d_model*2, d_model, bias=False, device=device)
        self.temporal_transformer_2 = Encoder(d_model, d_ff, heads_num, layers_num, dropout_rate, device)
        self.spatial_transformer_2 = Encoder(d_model, d_ff, heads_num, layers_num, dropout_rate, device)
        self.linear_dec1 = nn.Linear(d_model, emb_dim, bias=True, device=device)
        self.linear_dec2 = nn.Linear(emb_dim, output_dim, bias=True, device=device)
        self.device = device

    def forward(self, x, s_mask, t_mask, train_flg=True):
        agents, T = x.shape[0], x.shape[1]

        x = self.linear2(self.linear1(x))                                                                       # [agents, time_step, d_model]

        spatial_x, s_attention1 = self.spatial_transformer_1(x.transpose(0,1), s_mask, train_flg)               # [time_step, agents, d_model]
        temporal_x, t_attention1 = self.temporal_transformer_1(self.pos_encoding(x), t_mask, train_flg)         # [agents, time_step, d_model]
        x = torch.cat([spatial_x.transpose(0,1), temporal_x], dim=2)                                            # [agents, time_step, d_model*2]
        x = self.linear3(x)                                                                                     # [agents, time_step, d_model]

        x, t_attention2 = self.temporal_transformer_2(self.pos_encoding(x), t_mask, train_flg)                  # [agents, time_step, d_model]
        x, s_attention2 = self.spatial_transformer_2(x.transpose(0,1), s_mask, train_flg)                       # [time_step, agents, d_model]
        x = x.transpose(0,1)                                                                                    # [agents, time_step, d_model]

        if train_flg:
            x = x + torch.Tensor(np.random.normal(loc=0.0, scale=1.0, size=(agents, T, self.d_model))).to(self.device)

        output = self.linear_dec2(self.linear_dec1(x))                                                          # [agents, time_step, output_dim]
        output[:,:,[1,3]] = torch.exp(output[:,:,[1,3]])                                                  # 標準偏差のみ指数関数を通すことで負になるのを防ぐ

        return output, [t_attention1, t_attention2, s_attention1, s_attention2]  # [N, T, 4] 観察時間先T時刻のN人の(μx, σx, μy, σy)


# TSTS
class myModel3(nn.Module):
    def __init__(self, input_dim=4, output_dim=4, emb_dim=64, time_step=10, d_model=512, d_ff=2048, heads_num=8, layers_num=6, dropout_rate=0.1, device=torch.device('cpu')):
        super().__init__()
        self.emb_dim = emb_dim
        self.d_model = d_model
        self.time_step = time_step
        self.linear1 = nn.Linear(input_dim, emb_dim, bias=True, device=device)
        self.linear2 = nn.Linear(emb_dim, d_model, bias=True, device=device)
        self.pos_encoding = AddPositionalEncoding(d_model, time_step, device=device)
        self.temporal_transformer_1 = Encoder(d_model, d_ff, heads_num, layers_num, dropout_rate, device)
        self.spatial_transformer_1 = Encoder(d_model, d_ff, heads_num, layers_num, dropout_rate, device)
        self.temporal_transformer_2 = Encoder(d_model, d_ff, heads_num, layers_num, dropout_rate, device)
        self.spatial_transformer_2 = Encoder(d_model, d_ff, heads_num, layers_num, dropout_rate, device)
        self.linear_dec1 = nn.Linear(d_model, emb_dim, bias=True, device=device)
        self.linear_dec2 = nn.Linear(emb_dim, output_dim, bias=True, device=device)
        self.device = device

    def forward(self, x, s_mask, t_mask, train_flg=True):
        agents, T = x.shape[0], x.shape[1]

        x = self.linear2(self.linear1(x))                                                                      # [agents, time_step, d_model]

        x, t_attention1 = self.temporal_transformer_1(self.pos_encoding(x), t_mask, train_flg)                 # [agents, time_step, d_model]
        x, s_attention1 = self.spatial_transformer_1(x.transpose(0,1), s_mask, train_flg)                      # [time_step, agents, d_model]
        
        x, t_attention2 = self.temporal_transformer_2(self.pos_encoding(x.transpose(0,1)), t_mask, train_flg)  # [agents, time_step, d_model]
        x, s_attention2 = self.spatial_transformer_2(x.transpose(0,1), s_mask, train_flg)                      # [time_step, agents, d_model]
        x = x.transpose(0,1)                                                                                   # [agents, time_step, d_model]

        if train_flg:
            x = x + torch.Tensor(np.random.normal(loc=0.0, scale=1.0, size=(agents, T, self.d_model))).to(self.device)

        output = self.linear_dec2(self.linear_dec1(x))                                                         # [agents, time_step, output_dim]
        output[:,:,[1,3]] = torch.exp(output[:,:,[1,3]])                                                  # 標準偏差のみ指数関数を通すことで負になるのを防ぐ

        return output, [t_attention1, t_attention2, s_attention1, s_attention2]  # [N, T, 4] 観察時間先T時刻のN人の(μx, σx, μy, σy)



# STST
class myModel4(nn.Module):
    def __init__(self, input_dim=4, output_dim=4, emb_dim=64, time_step=10, d_model=512, d_ff=2048, heads_num=8, layers_num=6, dropout_rate=0.1, device=torch.device('cpu')):
        super().__init__()
        self.emb_dim = emb_dim
        self.d_model = d_model
        self.time_step = time_step
        self.linear1 = nn.Linear(input_dim, emb_dim, bias=True, device=device)
        self.linear2 = nn.Linear(emb_dim, d_model, bias=True, device=device)
        self.pos_encoding = AddPositionalEncoding(d_model, time_step, device=device)
        self.spatial_transformer_1 = Encoder(d_model, d_ff, heads_num, layers_num, dropout_rate, device)
        self.temporal_transformer_1 = Encoder(d_model, d_ff, heads_num, layers_num, dropout_rate, device)
        self.spatial_transformer_2 = Encoder(d_model, d_ff, heads_num, layers_num, dropout_rate, device)
        self.temporal_transformer_2 = Encoder(d_model, d_ff, heads_num, layers_num, dropout_rate, device)
        self.linear_dec1 = nn.Linear(d_model, emb_dim, bias=True, device=device)
        self.linear_dec2 = nn.Linear(emb_dim, output_dim, bias=True, device=device)
        self.device = device

    def forward(self, x, s_mask, t_mask, train_flg=True):
        agents, T = x.shape[0], x.shape[1]

        x = self.linear2(self.linear1(x))                                                                      # [agents, time_step, d_model]
        x, s_attention1 = self.spatial_transformer_1(x.transpose(0,1), s_mask, train_flg)                      # [time_step, agents, d_model]
        x, t_attention1 = self.temporal_transformer_1(self.pos_encoding(x.transpose(0,1)), t_mask, train_flg)  # [agents, time_step, d_model]
        x, s_attention2 = self.spatial_transformer_2(x.transpose(0,1), s_mask, train_flg)                      # [time_step, agents, d_model]
        x, t_attention2 = self.temporal_transformer_2(self.pos_encoding(x.transpose(0,1)), t_mask, train_flg)  # [agents, time_step, d_model]

        if train_flg:
            x = x + torch.Tensor(np.random.normal(loc=0.0, scale=1.0, size=(agents, T, self.d_model))).to(self.device)

        output = self.linear_dec2(self.linear_dec1(x))                                                         # [agents, time_step, output_dim]
        output[:,:,[1,3]] = torch.exp(output[:,:,[1,3]])                                                  # 標準偏差のみ指数関数を通すことで負になるのを防ぐ

        return output, [t_attention1, t_attention2, s_attention1, s_attention2]  # [N, T, 4] 観察時間先T時刻のN人の(μx, σx, μy, σy)