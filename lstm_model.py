import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from load_data_label import DataLoader
from create_dataset import DataProcessor
from loss_logger import LossLogger
from test_result_save import TestResultSaver
from training_logger import TrainingLogger  # ✅ 学習設定を保存するクラス
import numpy as np

# ✅ TensorFlow のデバッグメッセージを抑制
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ✅ 保存先ディレクトリを統一
save_dir = "test_results/LSTM_results"

# ✅ LossLogger のインスタンスを作成（保存パスを指定）
loss_logger = LossLogger(model_name="lstm_model", save_dir=save_dir)

# --- LSTM モデルの構築 ---
class LSTMModel(keras.Model):
    def __init__(self, num_units=128, num_layers=3, dropout_rate=0.3):
        super(LSTMModel, self).__init__()
        self.lstm_layers = []

        # 1層目の LSTM
        self.lstm_layers.append(layers.LSTM(
            num_units, return_sequences=True, activation="tanh",
            dropout=dropout_rate, recurrent_dropout=dropout_rate
        ))

        # 中間の LSTM 層
        for _ in range(num_layers - 2):
            self.lstm_layers.append(layers.LSTM(
                num_units, return_sequences=True, activation="tanh",
                dropout=dropout_rate, recurrent_dropout=dropout_rate
            ))

        # 最後の LSTM 層
        self.lstm_layers.append(layers.LSTM(
            num_units, return_sequences=False, activation="tanh",
            dropout=dropout_rate, recurrent_dropout=dropout_rate
        ))

        # 出力層
        self.output_layer = layers.Dense(6, activation="linear")

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.lstm_layers:
            x = layer(x, training=training)
        return self.output_layer(x)

# --- モデルのコンパイル ---
def build_lstm(input_shape):
    model = LSTMModel()
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.build(input_shape=(None,) + input_shape)  # ✅ 明示的に build()
    return model, optimizer

# --- メイン処理 ---
if __name__ == "__main__":
    print("=== データの作成を開始 ===")
    data_loader = DataLoader(data_dir="Data_Label/E420")
    x_data, y_label = data_loader.load_data()

    batch_size = 64
    learning_rate = 0.0001
    sample_input_shape = x_data.shape[1:]

    print("=== LSTM モデルの構築 ===")
    lstm_model, optimizer = build_lstm(sample_input_shape)

    # ✅ ダミーデータを流して output_shape を確定
    dummy_input = np.random.rand(1, *sample_input_shape).astype(np.float32)  # (1, シーケンス長, 特徴次元)
    lstm_model(dummy_input, training=False)  # ✅ モデルにデータを通して output_shape を確定

    # ✅ TrainingLogger を作成し、設定を保存
    training_logger = TrainingLogger(lstm_model, batch_size, learning_rate, optimizer, save_dir)
    training_logger.save_config()  # ✅ ここで `output_shape` は確定している

    print("=== モデルの学習を開始 ===")
    data_processor = DataProcessor(x_data, y_label, batch_size=batch_size)
    train_dataset, val_dataset, test_dataset = data_processor.get_datasets()

    lstm_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=100,
        callbacks=[loss_logger]
    )

    print("=== モデルの評価 ===")
    test_loss, test_mse = lstm_model.evaluate(test_dataset)
    print(f"Test Loss: {test_loss}, Test MSE: {test_mse}")

    print("=== モデルの保存 ===")
    lstm_model.save("lstm_model", save_format="tf")

    print("=== モデルの予測と保存を開始 ===")
    test_saver = TestResultSaver(save_dir=save_dir)
    test_saver.save_results(test_dataset, lstm_model, np.min(y_label), np.max(y_label))

    loss_logger.save_test_loss(test_loss)
    print("=== 学習ログが 'test_results/LSTM_results' に保存されました ===")
