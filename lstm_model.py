import os
import sys
import logging
import warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from load_data_label import DataLoader
from create_dataset import DataProcessor
from loss_logger import LossLogger
from test_result_save import TestResultSaver
from training_logger import TrainingLogger
import numpy as np

# ✅ TensorFlow のデバッグ情報を完全に抑制
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow のログレベルを ERROR のみに設定
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore")  # Python レベルの警告を無効化
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.debugging.set_log_device_placement(False)  # デバイス配置ログを抑制

# ✅ 保存先ディレクトリを統一
save_dir = "test_results/LSTM_results"
model_save_path = "lstm_model"
epochs = 50  # ✅ エポック数を変数化

# ✅ LossLogger のインスタンスを作成（保存パスを指定）
loss_logger = LossLogger(model_name="lstm_model", save_dir=save_dir)

# --- LSTM モデルの構築 ---
def build_lstm(input_shape, learning_rate):
    inputs = keras.Input(shape=input_shape)

    x = layers.LSTM(128, return_sequences=True, activation="tanh",
                    dropout=0.3, recurrent_dropout=0.3)(inputs)
    x = layers.BatchNormalization()(x)  # ✅ LSTM 層の出力にバッチ正規化を適用

    x = layers.LSTM(128, return_sequences=True, activation="tanh",
                    dropout=0.3, recurrent_dropout=0.3)(x)
    x = layers.BatchNormalization()(x)  # ✅ 2層目の LSTM の出力にも適用

    x = layers.LSTM(128, return_sequences=False, activation="tanh",
                    dropout=0.3, recurrent_dropout=0.3)(x)
    x = layers.BatchNormalization()(x)  # ✅ 最終 LSTM 層の後にも適用

    x = layers.Dense(6, activation="linear")(x)  # 出力層
    x = layers.BatchNormalization()(x)  # ✅ 出力層の前にも適用

    model = keras.Model(inputs, x)

    # ✅ 学習率を main から渡せるように修正
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mse"])
    
    return model, optimizer

# --- メイン処理 ---
if __name__ == "__main__":
    print("=== データの作成を開始 ===")
    data_loader = DataLoader(data_dir="Data_Label/Gym")
    x_data, y_label = data_loader.load_data()

    batch_size = 64
    learning_rate = 0.001  # ✅ ここで学習率を定義し、LSTM に渡す

    # ✅ x, y の最小値・最大値を取得
    x_min, x_max = np.min(x_data), np.max(x_data)
    y_min, y_max = np.min(y_label), np.max(y_label)

    sample_input_shape = x_data.shape[1:]

    print("=== LSTM モデルの構築 ===")
    # ✅ `learning_rate` を渡して LSTM の学習率を設定
    lstm_model, optimizer = build_lstm(sample_input_shape, learning_rate)

    # ✅ モデルを明示的に `build()` し、ダミーデータを通す
    lstm_model.build(input_shape=(None,) + sample_input_shape)
    dummy_input = np.zeros((1, 10, 6), dtype=np.float32)  # ✅ すべてゼロのダミーデータ
    lstm_model.predict(dummy_input)  # ✅ `predict()` で推論実行し `output_shape` を確定

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
        epochs=epochs,  # ✅ 変数を使う
        callbacks=[loss_logger]
    )

    print("=== モデルの評価 ===")
    test_loss, test_mse = lstm_model.evaluate(test_dataset)
    print(f"Test Loss: {test_loss}, Test MSE: {test_mse}")

    print("=== モデルの保存 ===")
    lstm_model.save(model_save_path, save_format="tf")

    print("=== モデルの予測と保存を開始 ===")
    test_saver = TestResultSaver(save_dir=save_dir)
    test_saver.save_results(test_dataset, lstm_model, np.min(y_label), np.max(y_label))

    loss_logger.save_test_loss(test_loss)
    print(f"=== 学習ログが '{save_dir}' に保存されました ===")