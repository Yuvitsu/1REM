import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from load_data_label import DataLoader
from create_dataset import DataProcessor
from loss_logger import LossLogger
from test_result_save import TestResultSaver  # ✅ 追加
from training_logger import TrainingLogger
import numpy as np

# ✅ 保存先ディレクトリを指定
save_dir = "test_results/Transformer_results"

# ✅ LossLogger のインスタンスを作成（保存パスを指定）
loss_logger = LossLogger(model_name="transformer_model", save_dir=save_dir)

# --- 位置エンコーディング ---
class PositionalEncoding(layers.Layer):
    def __init__(self, sequence_length, d_model):
        super(PositionalEncoding, self).__init__()
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.pos_encoding = self._get_positional_encoding()

    def _get_positional_encoding(self):
        positions = np.arange(self.sequence_length)[:, np.newaxis]
        i = np.arange(self.d_model)[np.newaxis, :]
        angles = positions / np.power(10000, (2 * (i // 2)) / np.float32(self.d_model))
        pos_encoding = np.zeros((self.sequence_length, self.d_model))
        pos_encoding[:, 0::2] = np.sin(angles[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(angles[:, 1::2])
        return tf.cast(pos_encoding[np.newaxis, ...], dtype=tf.float32)

    def call(self, x):
        return x + self.pos_encoding

# --- Transformer モデル ---
class TimeSeriesTransformer(keras.Model):
    def __init__(self, num_layers=2, d_model=128, num_heads=4, dff=512, dropout_rate=0.1, output_steps=6):
        super(TimeSeriesTransformer, self).__init__()

        # 入力変換
        self.input_layer = layers.Dense(d_model)

        # 位置エンコーディング
        self.pos_encoding = PositionalEncoding(sequence_length=10, d_model=d_model)

        # Transformer エンコーダ層
        self.encoder_layers = [
            layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model) for _ in range(num_layers)
        ]
        self.dropout_layers = [layers.Dropout(dropout_rate) for _ in range(num_layers)]
        self.norm_layers = [layers.LayerNormalization(epsilon=1e-6) for _ in range(num_layers)]

        # フィードフォワードネットワーク
        self.ffn_layers = [
            keras.Sequential([
                layers.Dense(dff, activation="relu"),
                layers.Dense(d_model)
            ]) for _ in range(num_layers)
        ]

        # 出力層
        self.output_layer = layers.Dense(output_steps, activation="linear")

    def call(self, inputs, training=False):
        x = self.input_layer(inputs)  # 次元変換
        x = self.pos_encoding(x)  # 位置エンコーディング追加

        for i in range(len(self.encoder_layers)):
            attn_output = self.encoder_layers[i](x, x, x)
            attn_output = self.dropout_layers[i](attn_output, training=training)
            x = self.norm_layers[i](x + attn_output)

            ffn_output = self.ffn_layers[i](x)
            ffn_output = self.dropout_layers[i](ffn_output, training=training)
            x = self.norm_layers[i](x + ffn_output)

        return self.output_layer(x[:, -1, :])  # 最後のタイムステップを出力

# --- モデルのコンパイル ---
def build_transformer():
    model = TimeSeriesTransformer()
    optimizer = keras.optimizers.Adam(learning_rate=0.001)  # ✅ 最適化関数を変数に保存
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mse"]
    )
    return model, optimizer  # ✅ 最適化関数を返す

# --- モデルの学習と損失の記録 ---
if __name__ == "__main__":
    print("=== データの作成を開始 ===")
    data_loader = DataLoader(data_dir="Data_Label/Gym")
    print("Data_Label/Gym")
    x_data, y_label = data_loader.load_data()

    # ✅ 学習時の x_data, y_label の min/max を取得
    x_min, x_max = np.min(x_data), np.max(x_data)
    y_min, y_max = np.min(y_label), np.max(y_label)

    # ✅ 正規化付きデータセットの作成（Min-Max 正規化）
    batch_size = 32
    learning_rate = 0.001
    data_processor = DataProcessor(x_data, y_label, batch_size=batch_size, normalization_method="minmax")
    train_dataset, val_dataset, test_dataset = data_processor.get_datasets()

    # ✅ モデルの構築
    transformer, optimizer = build_transformer()  # ✅ 最適化関数も取得

    # ✅ モデルを明示的に build() する（修正点）
    sample_input_shape = (None, 10, x_data.shape[-1])  # (batch_size, sequence_length, feature_dim)
    transformer.build(input_shape=sample_input_shape)

    # ✅ 設定を記録する `TrainingLogger` を作成し、設定を保存
    training_logger = TrainingLogger(transformer, batch_size, learning_rate, optimizer, x_min, x_max, y_min, y_max, save_dir)
    training_logger.save_config()

    print("=== モデルの学習を開始 ===")
    transformer.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=50,
        callbacks=[loss_logger]
    )

    print("=== モデルの評価 ===")
    test_loss, test_mse = transformer.evaluate(test_dataset)
    print(f"Test Loss: {test_loss}, Test MSE: {test_mse}")

    print("=== モデルの保存 ===")
    transformer.save("transformer_model", save_format="tf")

    # ✅ ここで `TestResultSaver` を使用して予測結果を保存
    print("=== モデルの予測と保存を開始 ===")
    test_saver = TestResultSaver(save_dir=save_dir)  # ✅ インスタンス作成
    test_saver.save_results(test_dataset, transformer, y_min, y_max)  # ✅ 結果を保存

    loss_logger.save_test_loss(test_loss)
    print(f"=== 学習ログが '{save_dir}' に保存されました ===")
