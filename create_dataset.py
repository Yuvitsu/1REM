import numpy as np
from load_data_label import DataLoader  # クラスをインポート
import tensorflow as tf

class DataProcessor:
    def __init__(self, x_data, y_label, batch_size=64, shuffle=True):
        """
        データを TensorFlow Dataset に変換するクラス

        Args:
            x_data (numpy.ndarray): 入力データ (形状: [サンプル数, 時系列長, 特徴量])
            y_label (numpy.ndarray): ラベルデータ (形状: [サンプル数, 出力次元数])
            batch_size (int): バッチサイズ
            shuffle (bool): シャッフルするかどうか（デフォルト: True）
        """
        self.x_data = x_data
        self.y_label = y_label
        self.batch_size = batch_size
        self.shuffle = shuffle

    def get_datasets(self, validation_split=0.2, test_split=0.1):
        """
        データセットをトレーニング、検証、テスト用に分割し、TensorFlow Dataset を返す

        Args:
            validation_split (float): 検証データの割合
            test_split (float): テストデータの割合
        
        Returns:
            train_dataset, val_dataset, test_dataset (tf.data.Dataset): 分割されたデータセット
        """
        total_samples = self.x_data.shape[0]
        val_size = int(total_samples * validation_split)
        test_size = int(total_samples * test_split)
        train_size = total_samples - val_size - test_size

        # データを分割
        x_train, y_train = self.x_data[:train_size], self.y_label[:train_size]
        x_val, y_val = self.x_data[train_size:train_size + val_size], self.y_label[train_size:train_size + val_size]
        x_test, y_test = self.x_data[train_size + val_size:], self.y_label[train_size + val_size:]

        # データを TensorFlow Dataset に変換
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        # シャッフル（train のみ）
        if self.shuffle:
            train_dataset = train_dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)

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

