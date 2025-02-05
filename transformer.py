import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from load_data_label import DataLoader
from create_dataset import DataProcessor
from loss_logger import LossLogger
import numpy as np

# ✅ TensorFlow のデバッグメッセージを抑制
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ✅ LossLogger のインスタンスを作成
loss_logger = LossLogger(train_log="train_loss.txt", val_log="val_loss.txt", test_log="test_loss.txt")

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

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

# --- Transformer モデル ---
class TransformerModel(keras.Model):
    def __init__(self, num_layers=3, d_model=128, num_heads=4, dff=512, dropout_rate=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = layers.Embedding(input_dim=10000, output_dim=d_model)
        self.pos_encoding = PositionalEncoding(sequence_length=100, d_model=d_model)
        self.encoder_layers = [
            layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
            for _ in range(num_layers)
        ]
        self.dense_layers = [
            layers.Dense(dff, activation="relu") for _ in range(num_layers)
        ]
        self.final_layer = layers.Dense(1)

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.pos_encoding(x)
        for mha, dense in zip(self.encoder_layers, self.dense_layers):
            attn_output = mha(x, x)
            x = layers.LayerNormalization()(x + attn_output)
            dense_output = dense(x)
            x = layers.LayerNormalization()(x + dense_output)
        return self.final_layer(x)

# --- モデルのトレーニング ---
def train_model():
    # ✅ データの準備
    data_loader = DataLoader()
    train_data, val_data, test_data = data_loader.load_data()

    # ✅ モデルの構築
    model = TransformerModel()
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    
    # ✅ コールバックとして損失を記録
    class LossHistoryCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            loss_logger.log_train_loss(logs['loss'])
            loss_logger.log_val_loss(logs['val_loss'])
    
    # ✅ トレーニング
    model.fit(
        train_data,
        epochs=20,
        validation_data=val_data,
        callbacks=[LossHistoryCallback()]
    )
    
    # ✅ モデルの保存
    model.save("transformer_model.h5")

if __name__ == "__main__":
    train_model()
