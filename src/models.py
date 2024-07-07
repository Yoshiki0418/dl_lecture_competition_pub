import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock2(hid_dim, hid_dim),
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
        X = self.blocks(X)
        X = X.permute(0, 2, 1) # RNNに渡すために形状を変更
        X = self.lstm_block(X)
        X = X.permute(0, 2, 1)

        return self.head(X)


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
    
class LSTM_Block(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            p_drop:float = 0.1,
            num_layers:int = 3,
            bidirectional: bool = False, #双方向にするか否か
    ) -> None:
        super().__init__()
        
        if(bidirectional):
          out_dim = int(out_dim/2)

        self.lstm = nn.LSTM(
            input_size = in_dim,
            hidden_size = out_dim,
            num_layers = num_layers,
            dropout = p_drop,
            batch_first=True,
            bidirectional=bidirectional
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
