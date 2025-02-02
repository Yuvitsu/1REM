import numpy as np
from load_data_label import DataLoader  # クラスをインポート
import tensorflow as tf

class DataProcessor:
    def __init__(self, x_data, y_label, batch_size=64, shuffle=True, normalization_method="minmax", validation_split=0.2, test_split=0.1):
        """
        データを TensorFlow Dataset に変換し、学習データの統計量を使って正規化を適用するクラス

        Args:
            x_data (numpy.ndarray): 入力データ (形状: [サンプル数, 時系列長, 特徴量])
            y_label (numpy.ndarray): ラベルデータ (形状: [サンプル数, 出力次元数])
            batch_size (int): バッチサイズ
            shuffle (bool): シャッフルするかどうか（デフォルト: True）
            normalization_method (str): "minmax" or "zscore"（デフォルト: minmax）
            validation_split (float): 検証データの割合
            test_split (float): テストデータの割合
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.normalization_method = normalization_method

        # データ分割
        total_samples = x_data.shape[0]
        indices = np.arange(total_samples)
        if shuffle:
            np.random.shuffle(indices)

        val_end = int(total_samples * validation_split)
        test_end = val_end + int(total_samples * test_split)

        train_indices = indices[test_end:]
        val_indices = indices[:val_end]
        test_indices = indices[val_end:test_end]

        # 学習データを抽出
        x_train = x_data[train_indices]
        y_train = y_label[train_indices]

        # 学習データの統計量を計算（この統計値を Val/Test にも適用）
        if normalization_method == "minmax":
            self.x_train, self.x_min, self.x_max = self.minmax_normalize(x_train)
            self.y_train, self.y_min, self.y_max = self.minmax_normalize(y_train)

            # Validation, Test にも学習データの統計値を適用
            self.x_val = self.apply_minmax(x_data[val_indices], self.x_min, self.x_max)
            self.y_val = self.apply_minmax(y_label[val_indices], self.y_min, self.y_max)

            self.x_test = self.apply_minmax(x_data[test_indices], self.x_min, self.x_max)
            self.y_test = self.apply_minmax(y_label[test_indices], self.y_min, self.y_max)

        elif normalization_method == "zscore":
            self.x_train, self.x_mean, self.x_std = self.zscore_normalize(x_train)
            self.y_train, self.y_mean, self.y_std = self.zscore_normalize(y_train)

            # Validation, Test にも学習データの統計値を適用
            self.x_val = self.apply_zscore(x_data[val_indices], self.x_mean, self.x_std)
            self.y_val = self.apply_zscore(y_label[val_indices], self.y_mean, self.y_std)

            self.x_test = self.apply_zscore(x_data[test_indices], self.x_mean, self.x_std)
            self.y_test = self.apply_zscore(y_label[test_indices], self.y_mean, self.y_std)

        else:
            raise ValueError("Invalid normalization method. Choose 'minmax' or 'zscore'.")

    def minmax_normalize(self, data):
        """Min-Max 正規化（学習データ用）"""
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1  # ゼロ割り防止
        normalized = (data - min_val) / range_val
        return normalized, min_val, max_val

    def apply_minmax(self, data, min_val, max_val):
        """Min-Max 正規化（学習データの統計値を使用）"""
        return (data - min_val) / (max_val - min_val)

    def zscore_normalize(self, data):
        """Z-score 標準化（学習データ用）"""
        mean_val = np.mean(data, axis=0)
        std_val = np.std(data, axis=0)
        std_val[std_val == 0] = 1  # ゼロ割り防止
        standardized = (data - mean_val) / std_val
        return standardized, mean_val, std_val

    def apply_zscore(self, data, mean_val, std_val):
        """Z-score 標準化（学習データの統計値を使用）"""
        return (data - mean_val) / std_val

    @staticmethod
    def minmax_denormalize(normalized_data, y_min, y_max):
        return normalized_data * (y_max - y_min) + y_min


    def zscore_denormalize(self, standardized_data, mean_val, std_val):
        """Z-score 逆標準化"""
        return standardized_data * std_val + mean_val

    def get_datasets(self):
        """
        TensorFlow Dataset を作成し、トレーニング、検証、テストデータを返す
        """
        # TensorFlow Dataset に変換
        train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        val_dataset = tf.data.Dataset.from_tensor_slices((self.x_val, self.y_val))
        test_dataset = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))

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
