import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from transformers import ViTFeatureExtractor, ViTModel
    
# CLIPモデル実装
class CLIP_model(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.vit_encoder = ViTEncoder(hid_dim, 281)
        self.eeg_encoder = EEGEncoder(in_channels, hid_dim)
    
    def forward(self, X: torch.Tensor, image: torch.Tensor):
        image_features = self.vit_encoder(image)
        eeg_features = self.eeg_encoder(X)
        
        # 対照学習の損失計算
        loss = self.contrastive_loss(image_features, eeg_features)
        
        return loss
    
    def contrastive_loss(self, image_features, eeg_features):
        # Flatten the features to [batch_size, features_dim]
        image_features_flat = image_features.reshape(image_features.size(0), -1)
        eeg_features_flat = eeg_features.reshape(eeg_features.size(0), -1)
        
        # コサイン類似度計算
        cos = nn.CosineSimilarity(dim=1)
        similarity = cos(image_features_flat, eeg_features_flat)
        
        # ラベルは正のペアのみを想定して1とします
        labels = torch.ones_like(similarity)
        
        # 損失計算 (コサイン類似度損失は1-similarityで計算されます)
        loss = F.mse_loss(similarity, labels)  # または別の適切な損失関数を使用
        return loss

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
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size) # , padding="same")
        
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

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)

        return self.dropout(X)
    
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

class ViTEncoder(nn.Module):
    def __init__(self, num_channels, sequence_length):
        super(ViTEncoder, self).__init__()
        self.num_channels = num_channels
        self.sequence_length = sequence_length
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        hidden_dim = self.model.config.hidden_size  # ViTの隠れ層の次元
        self.reshape_layer = nn.Linear(hidden_dim, num_channels * sequence_length)

    def forward(self, images):
        outputs = self.model(pixel_values=images)
        # CLSトークンの特徴を取得
        cls_features = outputs.last_hidden_state[:, 0]
        # 特徴次元を変換
        reshaped_output = self.reshape_layer(cls_features)
        # 最終的な出力を脳波データの形状に合わせる
        return reshaped_output.view(-1, self.num_channels, self.sequence_length)

    
class EEGEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        hid_dim,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.lstm_block = LSTM_Block(hid_dim, hid_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.blocks(X)
        # X = X.permute(0, 2, 1) # RNNに渡すために形状を変更
        # X = self.lstm_block(X)
        # X = X.permute(0, 2, 1)

        return X