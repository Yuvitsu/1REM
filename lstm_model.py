import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from load_data_label import DataLoader
from create_dataset import DataProcessor
from loss_logger import LossLogger
from test_result_save import TestResultSaver
from training_logger import TrainingLogger
import numpy as np

# ✅ TensorFlow のデバッグメッセージを抑制
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ✅ 保存先ディレクトリを統一
save_dir = "test_results/LSTM_results"

# ✅ LossLogger のインスタンスを作成（保存パスを指定）
loss_logger = LossLogger(model_name="lstm_model", save_dir=save_dir)

# --- LSTM モデルの構築 ---
def build_lstm(input_shape):
    inputs = keras.Input(shape=input_shape)

    x = layers.LSTM(128, return_sequences=True, activation="tanh",
                    dropout=0.3, recurrent_dropout=0.3)(inputs)
    x = layers.LSTM(128, return_sequences=True, activation="tanh",
                    dropout=0.3, recurrent_dropout=0.3)(x)
    x = layers.LSTM(128, return_sequences=False, activation="tanh",
                    dropout=0.3, recurrent_dropout=0.3)(x)

    outputs = layers.Dense(6, activation="linear")(x)
    model = keras.Model(inputs, outputs)

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mse"])
    
    return model, optimizer

# --- メイン処理 ---
if __name__ == "__main__":
    print("=== データの作成を開始 ===")
    data_loader = DataLoader(data_dir="Data_Label/E420")
    x_data, y_label = data_loader.load_data()

    batch_size = 64
    learning_rate = 0.0001

    # ✅ x, y の最小値・最大値を取得
    x_min, x_max = np.min(x_data), np.max(x_data)
    y_min, y_max = np.min(y_label), np.max(y_label)

    sample_input_shape = x_data.shape[1:]

    print("=== LSTM モデルの構築 ===")
    lstm_model, optimizer = build_lstm(sample_input_shape)

    # ✅ モデルを明示的に `build()` し、ダミーデータを通す
    lstm_model.build(input_shape=(None,) + sample_input_shape)
    dummy_input = np.random.rand(1, *sample_input_shape).astype(np.float32)  # (1, シーケンス長, 特徴次元)
    lstm_model(dummy_input, training=False)  # ✅ ここで `output_shape` を確定

    # ✅ `model.summary()` を実行して確実に `output_shape` を確定
    lstm_model.summary()

    # ✅ TrainingLogger を作成し、設定を保存
    training_logger = TrainingLogger(lstm_model, batch_size, learning_rate, optimizer, x_min, x_max, y_min, y_max, save_dir)
    training_logger.save_config()

    print("=== モデルの学習を開始 ===")
    data_processor = DataProcessor(x_data, y_label, batch_size=batch_size)
    train_dataset, val_dataset, test_dataset = data_processor.get_datasets()

    lstm_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=1,
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
