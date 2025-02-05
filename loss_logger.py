import tensorflow as tf

# ✅ Train Loss, Validation Loss, Test Loss を個別に保存するコールバック
class LossLogger(tf.keras.callbacks.Callback):
    def __init__(self, train_log="train_loss.txt", val_log="val_loss.txt", test_log="test_loss.txt"):
        super().__init__()
        self.train_log = train_log
        self.val_log = val_log
        self.test_log = test_log

        # 各ファイルの初期化（ヘッダーを書き込む）
        with open(self.train_log, "w") as f:
            f.write("Epoch,Train Loss\n")
        with open(self.val_log, "w") as f:
            f.write("Epoch,Val Loss\n")
        with open(self.test_log, "w") as f:
            f.write("Test Loss,Test MAE\n")  # Test のヘッダー

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

    def save_test_loss(self, test_loss, test_mae):
        """テストデータの Loss をファイルに保存"""
        with open(self.test_log, "a") as f:
            f.write(f"{test_loss},{test_mae}\n")
