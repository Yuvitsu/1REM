import numpy as np
from load_data_label import DataLoader  # クラスをインポート
import tensorflow as tf

import numpy as np
from load_data_label import DataLoader  # クラスをインポート
import tensorflow as tf

class DataProcessor:
    def __init__(self, x_data, y_label, batch_size=64, shuffle=True, normalization_method="minmax"):
        """
        データを TensorFlow Dataset に変換し、正規化を適用するクラス

        Args:
            x_data (numpy.ndarray): 入力データ (形状: [サンプル数, 時系列長, 特徴量])
            y_label (numpy.ndarray): ラベルデータ (形状: [サンプル数, 出力次元数])
            batch_size (int): バッチサイズ
            shuffle (bool): シャッフルするかどうか（デフォルト: True）
            normalization_method (str): "minmax" or "zscore"（デフォルト: minmax）
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.normalization_method = normalization_method

        # データの正規化
        if normalization_method == "minmax":
            self.x_train, self.x_min, self.x_max = self.minmax_normalize(x_data)
            self.y_train, self.y_min, self.y_max = self.minmax_normalize(y_label)
        elif normalization_method == "zscore":
            self.x_train, self.x_mean, self.x_std = self.zscore_normalize(x_data)
            self.y_train, self.y_mean, self.y_std = self.zscore_normalize(y_label)
        else:
            raise ValueError("Invalid normalization method. Choose 'minmax' or 'zscore'.")

    def minmax_normalize(self, data):
        """Min-Max 正規化"""
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        normalized = (data - min_val) / (max_val - min_val)
        return normalized, min_val, max_val

    def zscore_normalize(self, data):
        """Z-score 標準化"""
        mean_val = np.mean(data, axis=0)
        std_val = np.std(data, axis=0)
        standardized = (data - mean_val) / std_val
        return standardized, mean_val, std_val
    
    def minmax_denormalize(normalized_data, min_val, max_val):
        """Min-Max 逆正規化"""
        return normalized_data * (max_val - min_val) + min_val

    def zscore_denormalize(standardized_data, mean_val, std_val):
        """Z-score 逆標準化"""
        return standardized_data * std_val + mean_val


    def get_datasets(self, validation_split=0.2, test_split=0.1):
        """
        データセットをトレーニング、検証、テスト用に分割し、TensorFlow Dataset を返す
        """
        total_samples = self.x_train.shape[0]
        val_size = int(total_samples * validation_split)
        test_size = int(total_samples * test_split)
        train_size = total_samples - val_size - test_size

        # データを分割
        x_train, y_train = self.x_train[:train_size], self.y_train[:train_size]
        x_val, y_val = self.x_train[train_size:train_size + val_size], self.y_train[train_size:train_size + val_size]
        x_test, y_test = self.x_train[train_size + val_size:], self.y_train[train_size + val_size:]

        # TensorFlow Dataset に変換
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        # シャッフル（train のみ）
        if self.shuffle:
            train_dataset = train_dataset.shuffle(buffer_size=5000, reshuffle_each_iteration=True)

        # バッチ化
        train_dataset = train_dataset.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        val_dataset = val_dataset.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        test_dataset = test_dataset.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        return train_dataset, val_dataset, test_dataset

# --- ここからデータをロードし，データセットを作成する処理 ---
if __name__ == "__main__":
    # クラスをインスタンス化してデータをロード
    data_loader = DataLoader(data_dir="Data_Label/Gym")
    x_data, y_label = data_loader.load_data()

    # DataProcessor をインスタンス化してデータセットを作成
    data_processor = DataProcessor(x_data, y_label, batch_size=32)

    # データセットを取得
    train_dataset, val_dataset, test_dataset = data_processor.get_datasets()

    # 確認のため、1つ目のバッチを表示
    for x_batch, y_batch in train_dataset.take(1):
        print("\nSample training batch x:", x_batch.numpy().shape)
        print("Sample training batch y:", y_batch.numpy().shape)

