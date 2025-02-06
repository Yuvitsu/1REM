import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from load_data_label import DataLoader
from create_dataset import DataProcessor
from loss_logger import LossLogger
from test_result_save import TestResultSaver
import numpy as np

# ✅ TensorFlow のデバッグメッセージを抑制
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ✅ LossLogger のインスタンスを作成
loss_logger = LossLogger(model_name="lstm_model")

# --- LSTM モデルの構築 ---
class LSTMModel(keras.Model):
    def __init__(self, num_units=128, num_layers=3, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.lstm_layers = []

        # ✅ 1層目の LSTM
        self.lstm_layers.append(layers.LSTM(num_units, return_sequences=True, activation="tanh"))

        # ✅ 中間の LSTM 層
        for _ in range(num_layers - 2):
            self.lstm_layers.append(layers.LSTM(num_units, return_sequences=True, activation="tanh"))

        # ✅ 最後の LSTM 層（return_sequences=False）
        self.lstm_layers.append(layers.LSTM(num_units, return_sequences=False, activation="tanh"))

        # ✅ 出力層
        self.output_layer = layers.Dense(6, activation="linear")

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.lstm_layers:
            x = layer(x, training=training)
        return self.output_layer(x)

# --- モデルのコンパイル ---
def build_lstm(input_shape):
    model = LSTMModel()
    model.build(input_shape=(None,) + input_shape)
    
    # ✅ Optimizer を Adam に変更
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss="mse",  # ✅ 損失関数は MSE
        metrics=["mse"]  # ✅ 評価指標も MSE
    )
    return model

# --- メイン処理 ---
if __name__ == "__main__":
    print("=== データの作成を開始 ===")
    data_loader = DataLoader(data_dir="Data_Label/E420")
    print("Data_Label/E420")
    x_data, y_label = data_loader.load_data()
    print(np.min(x_data), np.max(x_data))
    print(np.min(y_label), np.max(y_label))

    # ✅ 学習時の x_data, y_label の min/max を取得
    x_min, x_max = np.min(x_data), np.max(x_data)
    y_min, y_max = np.min(y_label), np.max(y_label)

    print("=== データセットの作成を開始 ===")
    data_processor = DataProcessor(x_data, y_label, batch_size=32)
    train_dataset, val_dataset, test_dataset = data_processor.get_datasets()

    sample_input_shape = x_data.shape[1:]

    print("=== LSTM モデルの構築 ===")
    lstm_model = build_lstm(sample_input_shape)
    lstm_model.summary()

    print("=== モデルの学習を開始 ===")
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

    # ✅ テスト結果を保存するインスタンスを作成
    test_saver = TestResultSaver(save_dir="test_result/LSTM_test_results")

    # ✅ 修正: test_dataset, lstm_model, y_min, y_max を渡して処理
    test_saver.save_results(test_dataset, lstm_model, y_min, y_max)

    # ✅ LossLogger を使って Test Loss を記録
    loss_logger.save_test_loss(test_loss)
