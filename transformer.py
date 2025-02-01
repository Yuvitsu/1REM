import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from load_data_label import DataLoader
from create_dataset import DataProcessor

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
        metrics=["mae"]
    )
    return model

# --- データの作成 ---
def generate_dummy_data(num_samples=1000, time_steps=10, features=6):
    X = np.random.rand(num_samples, time_steps, features).astype(np.float32)
    y = np.random.rand(num_samples, features).astype(np.float32)  # 次の6つの値を予測
    return X, y

if __name__ == "__main__":
    # データ準備
    X_train, y_train = generate_dummy_data(num_samples=1000)
    X_val, y_val = generate_dummy_data(num_samples=200)

    # モデルを構築
    transformer = build_transformer()

    # モデルの学習
    transformer.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32
    )

    # モデルの評価
    test_loss, test_mae = transformer.evaluate(X_val, y_val)
    print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

    # **予測の実行**
    print("=== モデルによる予測開始 ===")

    # テストデータから1サンプルを取得
    x_test_sample = X_val[:5]  

    # 予測を実行
    predictions = transformer.predict(x_test_sample)

    # 予測結果を表示
    print("Actual y_test:", y_val[:5])  # 最初の5つのラベル
    print("Predicted y:", predictions[:5])  # 最初の5つの予測結果
