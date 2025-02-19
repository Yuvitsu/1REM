import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from load_data_label import DataLoader
from create_dataset import DataProcessor
from loss_logger import LossLogger
from test_result_save import TestResultSaver
from training_summary_saver import TrainingSummarySaver

# ✅ TensorFlow のデバッグメッセージを抑制
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

# --- 深い Transformer モデル ---
class DeepTimeSeriesTransformer(keras.Model):
    def __init__(self, num_layers=6, d_model=256, num_heads=8, dff=1024, dropout_rate=0.2, output_steps=6):
        super(DeepTimeSeriesTransformer, self).__init__()

        # 入力変換層
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

# --- Transformer のトレーニングクラス ---
class DeepTransformerTrainer:
    def __init__(self, data_dir="Data_Label/Gym", batch_size=32, epochs=50):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.epochs = epochs

        # ✅ LossLogger のインスタンスを作成
        self.loss_logger = LossLogger(model_name="deep_transformer_model")

        # ✅ 学習情報保存クラスのインスタンス
        self.summary_saver = TrainingSummarySaver()

        # ✅ データのロード
        self.data_loader = DataLoader(data_dir=self.data_dir)
        self.x_data, self.y_label = self.data_loader.load_data()

        # ✅ データの正規化
        self.data_processor = DataProcessor(self.x_data, self.y_label, batch_size=self.batch_size, normalization_method="minmax")
        self.train_dataset, self.val_dataset, self.test_dataset = self.data_processor.get_datasets()

        # ✅ Transformer モデルを構築
        self.model = self.build_model()

    def build_model(self):
        model = DeepTimeSeriesTransformer(num_layers=6, d_model=256, num_heads=8, dff=1024)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss="mse",
            metrics=["mse"]
        )
        return model

    def train(self):
        print("=== モデルの学習を開始 ===")
        history = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=self.epochs,
            callbacks=[self.loss_logger]
        )

        # ✅ 最終エポックの MSE を取得
        final_train_mse = history.history['mse'][-1]
        final_val_mse = history.history['val_mse'][-1]

        # ✅ モデルの評価
        test_loss, test_mse = self.model.evaluate(self.test_dataset)

        # ✅ 学習情報を保存
        self.summary_saver.save_summary(self.batch_size, final_train_mse, final_val_mse, test_mse, self.model)

        # ✅ **学習後のモデルを保存**
        self.model.save("deep_transformer_model", save_format="tf")

        print(f"Test Loss: {test_loss}, Test MSE: {test_mse}")

        return test_loss, test_mse

    def predict_and_save_results(self):
        print("=== モデルの予測と保存を開始 ===")
        test_saver = TestResultSaver(save_dir="test_results/Deep_Transformer_test_results")
        test_saver.save_results(self.test_dataset, self.model, np.min(self.y_label), np.max(self.y_label))
        print("=== 予測結果を保存しました ===")

# --- メイン処理 ---
if __name__ == "__main__":
    trainer = DeepTransformerTrainer(data_dir="Data_Label/Gym", batch_size=32, epochs=50)
    trainer.train()
    trainer.predict_and_save_results()
