import time
import tensorflow as tf  # ✅ 追加
from tensorflow import keras  # ✅ 追加

# ✅ 学習プロセス全体の情報をログに記録
class TrainingLogger(keras.callbacks.Callback):
    def __init__(self, model, batch_size, learning_rate, optimizer, save_path="training_summary.txt"):
        super(TrainingLogger, self).__init__()
        self.save_path = save_path
        self.model = model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.start_time = None
        self.min_train_loss = float("inf")
        self.min_val_loss = float("inf")
        self.final_train_loss = None
        self.final_val_loss = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_loss = logs.get("mse")  # 訓練データの MSE
        val_loss = logs.get("val_mse")  # 検証データの MSE

        # 最小損失を更新
        if train_loss is not None and train_loss < self.min_train_loss:
            self.min_train_loss = train_loss
        if val_loss is not None and val_loss < self.min_val_loss:
            self.min_val_loss = val_loss

        # 最終エポックの損失を保存
        self.final_train_loss = train_loss
        self.final_val_loss = val_loss

    def on_train_end(self, logs=None):
        end_time = time.time()
        training_time = end_time - self.start_time

        # モデルの層構造を取得
        model_structure = []
        for layer in self.model.layers:
            model_structure.append(f"{layer.__class__.__name__}: {layer.output_shape}")

        # ログをファイルに保存
        with open(self.save_path, "w") as f:
            f.write("=== 学習ログ ===\n")
            f.write(f"バッチサイズ: {self.batch_size}\n")
            f.write(f"学習率: {self.learning_rate}\n")
            f.write(f"最適化関数: {self.optimizer.__class__.__name__}\n")
            f.write("\n=== モデル構造 ===\n")
            f.write("\n".join(model_structure) + "\n")
            f.write("\n=== 学習結果 ===\n")
            f.write(f"最終エポックの 訓練 MSE: {self.final_train_loss:.6f}\n")
            f.write(f"最終エポックの 検証 MSE: {self.final_val_loss:.6f}\n")
            f.write(f"最小 訓練 MSE: {self.min_train_loss:.6f}\n")
            f.write(f"最小 検証 MSE: {self.min_val_loss:.6f}\n")
            f.write(f"学習時間: {training_time:.2f} 秒\n")
