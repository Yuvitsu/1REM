import os
import tensorflow as tf

# ✅ Train Loss, Validation Loss, Test Loss を個別に保存するコールバック
class LossLogger(tf.keras.callbacks.Callback):
    def __init__(self, model_name="default_model", save_dir="test_results"):
        super().__init__()
        self.log_dir = os.path.join(save_dir, model_name)  # ✅ 指定されたパスに保存
        os.makedirs(self.log_dir, exist_ok=True)  # ディレクトリを作成

        self.train_log = os.path.join(self.log_dir, "train_loss.txt")
        self.val_log = os.path.join(self.log_dir, "val_loss.txt")
        self.test_log = os.path.join(self.log_dir, "test_loss.txt")

        # 各ファイルの初期化（ヘッダーを書き込む）
        with open(self.train_log, "w") as f:
            f.write("Epoch,Train Loss\n")
        with open(self.val_log, "w") as f:
            f.write("Epoch,Val Loss\n")
        with open(self.test_log, "w") as f:
            f.write("Test Loss\n")  # Test のヘッダー

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_loss = logs.get("loss", float("nan"))
        val_loss = logs.get("val_loss", float("nan"))

        # Train Loss の保存
        with open(self.train_log, "a") as f:
            f.write(f"{epoch+1},{train_loss}\n")

        # Validation Loss の保存
        with open(self.val_log, "a") as f:
            f.write(f"{epoch+1},{val_loss}\n")

    def save_test_loss(self, test_loss):
        """テストデータの Loss をファイルに保存"""
        with open(self.test_log, "a") as f:
            f.write(f"{test_loss}\n")
