import numpy as np
import os
"""
DataLoader クラス

このクラスは、指定したディレクトリ内にある `x_data.npy` と `y_label.npy` の NumPy ファイルを
読み込み、データの形状を表示した後に、NumPy 配列として返す。

主な機能:
- 指定されたディレクトリから `x_data.npy` と `y_label.npy` をロード
- ファイルの存在をチェックし、見つからない場合は `FileNotFoundError` を発生
- 読み込んだデータの形状を表示
- `load_data()` メソッドでデータを取得し、`x_data, y_label` のタプルとして返す

使用例:
```python
data_loader = DataLoader(data_dir="Data_Label/Gym")  # インスタンス作成
x_data, y_label = data_loader.load_data()  # データを取得
numpyはfloat64なので，float32にして最終的に出力します．上田
"""
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
        x_data = np.load(x_path).astype(np.float32)  # 🔹 float32 に変換
        y_label = np.load(y_path).astype(np.float32)  # 🔹 float32 に変換

        # 形状を表示
        print("x_data shape:", x_data.shape, "dtype:", x_data.dtype)
        print("y_label shape:", y_label.shape, "dtype:", y_label.dtype)


        return x_data, y_label

# --- ここからデータをロードする処理 ---
if __name__ == "__main__":
    # クラスをインスタンス化
    data_loader = DataLoader(data_dir="Data_Label/Gym")

    # データをロード
    x_data, y_label = data_loader.load_data()

    
