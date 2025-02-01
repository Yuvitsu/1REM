import numpy as np
from load_data_label import DataLoader  # クラスをインポート
import tensorflow as tf

class DataProcessor:
    def __init__(self, x_data, y_label, batch_size=32, seed=42):
        """
        既存の x_data と y_label を受け取り、データセットを分割して TensorFlow Dataset に変換するクラス

        Args:
            x_data (np.array): 特徴量データ (例: (31726, 10, 6))
            y_label (np.array): ラベルデータ (例: (31726, 6))
            batch_size (int): バッチサイズ (デフォルト: 32)
            seed (int): シャッフル時のシード値 (デフォルト: 42)
        """
        self.x_data = x_data
        self.y_label = y_label
        self.batch_size = batch_size
        self.seed = seed
        self.dataset_size = x_data.shape[0]

        # デバッグ用出力: 初期データの形状
        print("=== DataProcessor Initialized ===")
        print(f"Total dataset size: {self.dataset_size}")
        print(f"x_data shape: {self.x_data.shape}, y_label shape: {self.y_label.shape}")
        print(f"Seed value: {self.seed}")

        # データ分割処理
        self.split_data()

        # TensorFlow Dataset を作成
        self.create_datasets()

    def split_data(self):
        """
        データセットを 7:2:1 にシャッフル & 分割
        """
        train_size = int(self.dataset_size * 0.7)  # 70% training
        val_size = int(self.dataset_size * 0.2)    # 20% validation
        test_size = self.dataset_size - (train_size + val_size)  # 10% test

        # シャッフル
        np.random.seed(self.seed)
        indices = np.random.permutation(self.dataset_size)

        # デバッグ用: シャッフル後の最初の10個のインデックスを表示
        print("\n=== Shuffling Done ===")
        print(f"First 10 shuffled indices: {indices[:10]}")

        # シャッフルしたデータを分割
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        self.x_train, self.y_train = self.x_data[train_indices], self.y_label[train_indices]
        self.x_val, self.y_val = self.x_data[val_indices], self.y_label[val_indices]
        self.x_test, self.y_test = self.x_data[test_indices], self.y_label[test_indices]

        # デバッグ用出力: 分割後のデータの形状
        print("\n=== Data Splitting Completed ===")
        print(f"Training dataset size: {len(self.x_train)}")
        print(f"Validation dataset size: {len(self.x_val)}")
        print(f"Test dataset size: {len(self.x_test)}")

        # デバッグ用出力: 最初の1サンプルを表示
        print(f"\nFirst training sample x:\n{self.x_train[0]}")
        print(f"First training sample y:\n{self.y_train[0]}")

    def create_datasets(self):
        """
        NumPy 配列を TensorFlow Dataset に変換
        """
        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train)).shuffle(1000).batch(self.batch_size)
        self.val_dataset = tf.data.Dataset.from_tensor_slices((self.x_val, self.y_val)).batch(self.batch_size)
        self.test_dataset = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test)).batch(self.batch_size)

        # デバッグ用出力: Dataset の情報
        print("\n=== TensorFlow Dataset Created ===")
        print(f"Train dataset: {self.train_dataset}")
        print(f"Validation dataset: {self.val_dataset}")
        print(f"Test dataset: {self.test_dataset}")

    def get_datasets(self):
        """
        Dataset を取得するメソッド
        """
        return self.train_dataset, self.val_dataset, self.test_dataset


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

