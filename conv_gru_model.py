import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import gc
from load_data_label import DataLoader
from create_dataset import DataProcessor
from rssi_interpolator import RSSIInterpolator
from loss_logger import LossLogger
from test_result_save import TestResultSaver
from tf_dataset_builder import TFDatasetBuilder

# ✅ TensorFlow のデバッグメッセージを抑制
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ✅ メモリを解放してからモデルを作成
K.clear_session()
gc.collect()

# ✅ LossLogger のインスタンスを作成
loss_logger = LossLogger(model_name="conv_lstm_model")

# --- ConvLSTM モデルの構築 ---
def build_conv_lstm(input_shape):
    model = keras.Sequential([
        # ✅ ConvLSTM2D 層
        layers.ConvLSTM2D(filters=64, kernel_size=(4, 4), padding="same", return_sequences=False, activation="tanh", input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # ✅ Conv層1（フィルター数 32）
        layers.Conv2D(filters=32, kernel_size=(4, 4), activation="tanh", padding="same"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # ✅ Conv層2（フィルター数 16）
        layers.Conv2D(filters=16, kernel_size=(4, 4), activation="tanh", padding="same"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # ✅ Conv層3（フィルター数 8）
        layers.Conv2D(filters=8, kernel_size=(4, 4), activation="tanh", padding="same"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # ✅ Conv層4（出力層: 1フィルター, Linear）
        layers.Conv2D(filters=1, kernel_size=(4, 4), activation="linear", padding="same")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss="mse",
        metrics=["mse"]
    )
    return model

# --- メイン処理 ---
if __name__ == "__main__":
    print("=== データの作成を開始 ===")
    data_loader = DataLoader(data_dir="Data_Label/Gym")
    x_data, y_label = data_loader.load_data()

    # ✅ 学習時の x_data, y_label の min/max を取得
    x_min, x_max = np.min(x_data), np.max(x_data)
    y_min, y_max = np.min(y_label), np.max(y_label)

    print("=== スプライン補間を適用 ===")
    interpolator = RSSIInterpolator(grid_size=(100, 100))
    x_data_interp = np.array([interpolator.interpolate(sample) for sample in x_data])  # (サンプル数, 100, 100, 10, 6)
    y_label_interp = np.array([interpolator.interpolate(sample) for sample in y_label])  # (サンプル数, 100, 100)

    # ✅ x_data_interp にチャンネル次元を追加 (5D テンソルにする)
    x_data_interp = x_data_interp[..., np.newaxis]  # (31513, 10, 100, 100, 1)

    # ✅ y_label の形状を (サンプル数, 100, 100, 1) に変換
    y_label_interp = y_label_interp.reshape(-1, 100, 100, 1)

    print("x_data_interp.shape:", x_data_interp.shape)  # 期待: (31513, 10, 100, 100, 1)
    print("y_label_interp.shape:", y_label_interp.shape)  # 期待: (31513, 100, 100, 1)

    print("=== データセットの作成を開始 ===")
    data_processor = DataProcessor(x_data_interp, y_label_interp, batch_size=16, normalization_method="minmax")
    train_dataset, val_dataset, test_dataset = data_processor.get_datasets()

    # ✅ 不要なデータを削除し、メモリを解放
    del x_data, y_label, x_data_interp, y_label_interp
    gc.collect()

    # ✅ sample_input_shape を明示的に設定
    time_steps = 10
    sample_input_shape = (time_steps, 100, 100, 1)
    print("sample_input_shape", sample_input_shape)

    print("=== ConvLSTM モデルの構築 ===")
    conv_lstm_model = build_conv_lstm(sample_input_shape)
    conv_lstm_model.summary()

    print("=== モデルの学習を開始 ===")
    conv_lstm_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=50,
        callbacks=[loss_logger]
    )

    print("=== モデルの評価 ===")
    test_loss, test_mse = conv_lstm_model.evaluate(test_dataset)
    print(f"Test Loss: {test_loss}, Test MSE: {test_mse}")

    print("=== モデルの保存 ===")
    conv_lstm_model.save("conv_lstm_model", save_format="tf")

    print("=== モデルの予測と保存を開始 ===")

    # ✅ テスト結果を保存するインスタンスを作成
    test_saver = TestResultSaver(save_dir="test_results")

    # ✅ 予測結果の保存
    test_saver.save_results(test_dataset, conv_lstm_model, y_min, y_max)

    # ✅ LossLogger を使って Test Loss を記録
    loss_logger.save_test_loss(test_loss)

    # ✅ 学習後のメモリ解放
    K.clear_session()
    gc.collect()
