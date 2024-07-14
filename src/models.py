import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from einops import rearrange
from torch import dropout, dropout_, einsum

#------------------------------
# モデル
#------------------------------
class MEG_transformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128,
        dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.adjust_dim = nn.Conv1d(271, 272, kernel_size=1)
        self.pe = PositionalEncoding(dim=in_channels)

        self.blocks = nn.Sequential(
            TransformerBlock(in_channels, seq_len, dropout),
            nn.LayerNorm([in_channels, seq_len]),  # Normalize after first Transformer
            TransformerBlock(in_channels, seq_len, dropout),
            nn.LayerNorm([in_channels, seq_len]),  # Normalize after second Transformer
            ConvBlock(in_channels, hid_dim),
        )

        self.lstm_block = LSTM_Block(hid_dim, hid_dim)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        time = torch.arange(X.shape[-1], device=X.device)
        time_emb = self.pe(time=time)
        time_emb = time_emb.transpose(0, 1).to(X.device)
        X = X + time_emb.unsqueeze(0)
        X = self.blocks(X)

        return self.head(X)

#------------------------------
#CNN1
#------------------------------
class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size+2, padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        return self.dropout(X)
    
#------------------------------
#CNN2
#------------------------------
class ConvBlock2(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 7,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size-4, padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        return self.dropout(X)

#------------------------------
#LSTM
#------------------------------ 
class LSTM_Block(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            p_drop:float = 0.1,
            num_layers:int = 3,
            bidirectional: bool = True, #双方向にするか否か
    ) -> None:
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size = in_dim,
            hidden_size = out_dim,
            num_layers = num_layers,
            dropout = p_drop,
            batch_first=True
        )

        self.dropout = nn.Dropout(p_drop)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        入力Xの形状：(batch_size, seq_len, features)
        seq_len : シーケンス長（時系列データの長さ）
        features : 各時点に関連する特徴の数
        """
        X, (hn, cn) = self.lstm(X)
        X = self.dropout(X)
        return X

#------------------------------
#ファインチューニングモデル
#------------------------------
class FineTunedCLIPModel(nn.Module):
    def __init__(
            self,
            num_classes,
            pretrained_model,
            hid_dim:int = 128,
        ):
        super().__init__()
    
        self.pretrained_model = pretrained_model
        # 事前学習済みの特徴抽出部分のパラメータを固定
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        # 新たな分類用ヘッドを追加
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, x):
        features = self.pretrained_model.eeg_encoder(x)  # 脳波データから特徴を抽出
        output = self.head(features) # 新たなヘッドでクラス予測
        return output

#------------------------------
#絶対位置エンコーディング
#------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
 
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        pe = math.log(10000) / (half_dim - 1) # pe => positional_encoding
        pe = torch.exp(torch.arange(half_dim, device=device) * -pe)
        pe = time[:, None] * pe[None, :]
        pe = torch.cat((pe.sin(), pe.cos()), dim=-1)
        if pe.size(1) < self.dim:
            zero_padding = torch.zeros(pe.size(0), 1, device=device)
            pe = torch.cat([pe, zero_padding], dim=1)
        return pe

#------------------------------
#Feed Forward(MLP)
#------------------------------
class FFN(nn.Module):
    def __init__(self, dim, dff=240, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(dim, dim*2)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(dim*2, dim)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x

#------------------------------
#Multi-Head Self-Attention
#------------------------------
class Attention(nn.Module):
    """
    dim : int
        入力データの次元数．埋め込み次元数と一致する．
    heads : int
        ヘッドの数．
    dim_head : int
        各ヘッドのデータの次元数．
     dropout : float
        Dropoutの確率(default=0.)．
    """
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** (-0.5)
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, l = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1) # 線形変換
        q, k, v = [rearrange(t, "b (h d) l -> b h d l", h=self.heads) for t in qkv]
        q = q * self.scale
        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h l d -> b (h d) l", l=l)
        out = self.to_out(out)
        return out

#------------------------------
#Transformer
#------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, length, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm((dim, length))
        self.attn = Attention(dim)
        self.ffn = FFN(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_norm1 = self.norm(x)
        x_norm1 = self.attn(x_norm1)
        x2 = x + self.dropout(x_norm1)
        x_norm2 = self.norm(x2)
        x_norm2 = rearrange(x_norm2, "b c l -> b l c")
        x_norm2 = self.ffn(x_norm2)
        x_norm2 = self.dropout(x_norm2)
        x_norm2 = rearrange(x_norm2, "b l c -> b c l")
        output = x2 + x_norm2
        return output
