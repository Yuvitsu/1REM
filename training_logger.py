import os
import tensorflow as tf
from tensorflow import keras

class TrainingLogger:
    def __init__(self, model, batch_size, learning_rate, optimizer, x_min, x_max, y_min, y_max, save_dir="test_results/Transformer_results"):
        self.model = model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer  # ✅ 最適化関数を追加
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.save_path = os.path.join(save_dir, "training_config.txt")

        # 出力ディレクトリが存在しない場合は作成
        os.makedirs(save_dir, exist_ok=True)

    def save_config(self):
        """学習設定（学習率・バッチサイズ・最適化関数・層の構造・データ統計）をテキストファイルに保存"""
        with open(self.save_path, "w") as f:
            f.write("=== 学習設定 ===\n")
            f.write(f"バッチサイズ: {self.batch_size}\n")
            f.write(f"学習率: {self.learning_rate}\n")
            f.write(f"最適化関数: {self.optimizer.__class__.__name__}\n")  # ✅ 最適化関数の名前を記録
            f.write("\n=== モデル構造 ===\n")

            # 各層の構造を保存
            for layer in self.model.layers:
                layer_output_shape = getattr(layer, 'output_shape', '未定義')  # ✅ `getattr()` で `AttributeError` を回避
                layer_info = f"{layer.__class__.__name__}: {layer_output_shape}"
                f.write(layer_info + "\n")

            # ✅ x, y の最小・最大値を記録
            f.write("\n=== データ統計 ===\n")
            f.write(f"x_min: {self.x_min:.6f}, x_max: {self.x_max:.6f}\n")
            f.write(f"y_min: {self.y_min:.6f}, y_max: {self.y_max:.6f}\n")
