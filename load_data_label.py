import numpy as np
import os

class DataLoader:
    def __init__(self, data_dir="Data_Label/Gym"):
        """
        初期化: データディレクトリを指定
        """
        self.data_dir = data_dir  # データが保存されているディレクトリ

    def load_data(self):
        """
        x_data.npy と y_label.npy を読み込み、形状を表示
        """
        x_path = os.path.join(self.data_dir, "x_data.npy")
        y_path = os.path.join(self.data_dir, "y_label.npy")

        # ファイルが存在するかチェック
        if not os.path.exists(x_path) or not os.path.exists(y_path):
            raise FileNotFoundError(f"データファイルが見つかりません: {x_path} または {y_path}")

        # NumPy ファイルを読み込む
        x_data = np.load(x_path)
        y_label = np.load(y_path)

        # 形状を表示
        print("x_data shape:", x_data.shape)
        print("y_label shape:", y_label.shape)

        return x_data, y_label

# --- ここからデータをロードする処理 ---
if __name__ == "__main__":
    # クラスをインスタンス化
    data_loader = DataLoader(data_dir="Data_Label/Gym")

    # データをロード
    x_data, y_label = data_loader.load_data()

    
