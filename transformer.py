import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from load_data_label import DataLoader
from create_dataset import DataProcessor
from loss_logger import LossLogger
import numpy as np

# ✅ LossLogger のインスタンスを作成（ディレクトリごとに保存可能）
loss_logger = LossLogger(model_name="transformer_model")

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
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mse"]
    )
    return model

# --- モデルの学習と損失の記録 ---
if __name__ == "__main__":
    # データのロード
    #data_loader = DataLoader(data_dir="Data_Label/Gym")
    # print("Data_Label/Gym")
    data_loader = DataLoader(data_dir="Data_Label/E420") # E420データセットの時はこれ
    print("Data_Label/E420")
    x_data, y_label = data_loader.load_data()



    # 正規化付きデータセットの作成
    data_processor = DataProcessor(x_data, y_label, batch_size=32, normalization_method="minmax")  # Min-Max正規化
    train_dataset, val_dataset, test_dataset = data_processor.get_datasets()

    # モデルの構築
    transformer = build_transformer()
    
    # ✅ コールバックとして損失を記録
    class LossHistoryCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            loss_logger.log_train_loss(logs.get("loss", float("nan")))
            loss_logger.log_val_loss(logs.get("val_loss", float("nan")))
    
    # ✅ 学習
    history = transformer.fit(train_dataset, validation_data=val_dataset, epochs=100, callbacks=[loss_logger])
    
    # ✅ テストデータの損失記録（MSE のみを保存）
    test_loss, test_mse = transformer.evaluate(test_dataset)
    loss_logger.save_test_loss(test_loss)  # 修正：MSE のみを保存
    
    # ✅ 予測処理
    test_iter = iter(test_dataset)
    x_test_sample, y_test_sample = next(test_iter)
    predictions_normalized = transformer.predict(x_test_sample)

    # 逆正規化
    predictions_original = DataProcessor.minmax_denormalize(predictions_normalized, data_processor.y_min, data_processor.y_max)
    actual_original = DataProcessor.minmax_denormalize(y_test_sample.numpy(), data_processor.y_min, data_processor.y_max)
    
    # データの最大値と最小値を表示
    print("x_data max:", data_processor.x_max)
    print("x_data min:", data_processor.x_min)
    print("y_label max:", data_processor.y_max)
    print("y_label min:", data_processor.y_min)


    # 予測結果の表示
    print("\n=== 予測結果（元のスケール） ===")
    print("Actual y_test (Original Scale):", actual_original[:5])  
    print("Predicted y (Original Scale):", predictions_original[:5])
