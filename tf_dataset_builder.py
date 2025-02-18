import tensorflow as tf
import numpy as np
from load_data_label import DataLoader

class TFDatasetBuilder:
    def __init__(self, x_data, y_data, batch_size=32, shuffle=True, normalization_method="minmax", prefetch=True):
        """
        TensorFlow Dataset を最適化するクラス

        Args:
            x_data (numpy.ndarray): 入力データ (形状: [サンプル数, 高さ, 幅, チャンネル数])
            y_data (numpy.ndarray): ラベルデータ (形状: [サンプル数, 高さ, 幅, 1])
            batch_size (int): バッチサイズ（デフォルト: 64）
            shuffle (bool): データをシャッフルするかどうか（デフォルト: True）
            normalization_method (str): "minmax" または "zscore"（デフォルト: minmax）
            prefetch (bool): `tf.data.experimental.AUTOTUNE` を適用するかどうか
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.prefetch = prefetch
        self.normalization_method = normalization_method

        # データの正規化
        if normalization_method == "minmax":
            self.x_data, self.x_min, self.x_max = self.minmax_normalize(x_data)
            self.y_data, self.y_min, self.y_max = self.minmax_normalize(y_data)
        elif normalization_method == "zscore":
            self.x_data, self.x_mean, self.x_std = self.zscore_normalize(x_data)
            self.y_data, self.y_mean, self.y_std = self.zscore_normalize(y_data)
        else:
            raise ValueError("Invalid normalization method. Choose 'minmax' or 'zscore'.")

    def minmax_normalize(self, data):
        """Min-Max 正規化"""
        min_val = np.min(data)
        max_val = np.max(data)
        range_val = max_val - min_val
        if range_val == 0:
            range_val = 1
        normalized = (data - min_val) / range_val
        return normalized, min_val, max_val

    def zscore_normalize(self, data):
        """Z-score 正規化"""
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            std_val = 1
        standardized = (data - mean_val) / std_val
        return standardized, mean_val, std_val

    def create_tf_dataset(self):
        """tf.data.Dataset を作成し、最適化する"""
        dataset = tf.data.Dataset.from_tensor_slices((self.x_data, self.y_data))

        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=5000, reshuffle_each_iteration=True)

        dataset = dataset.batch(self.batch_size)

        if self.prefetch:
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def get_normalization_params(self):
        """正規化パラメータを取得"""
        if self.normalization_method == "minmax":
            return {"x_min": self.x_min, "x_max": self.x_max, "y_min": self.y_min, "y_max": self.y_max}
        elif self.normalization_method == "zscore":
            return {"x_mean": self.x_mean, "x_std": self.x_std, "y_mean": self.y_mean, "y_std": self.y_std}

    def denormalize(self, normalized_data, target="y"):
        """正規化を元のスケールに戻す"""
        if target == "x":
            if self.normalization_method == "minmax":
                return normalized_data * (self.x_max - self.x_min) + self.x_min
            else:
                return normalized_data * self.x_std + self.x_mean
        elif target == "y":
            if self.normalization_method == "minmax":
                return normalized_data * (self.y_max - self.y_min) + self.y_min
            else:
                return normalized_data * self.y_std + self.y_mean
        else:
            raise ValueError("Target must be 'x' or 'y'")
