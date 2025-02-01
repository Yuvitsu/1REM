import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from load_data_label import DataLoader
from create_dataset import DataProcessor

# --- Transformer モデルの構築 ---
class TransformerModel(keras.Model):
    def __init__(self, num_layers=2, d_model=128, num_heads=4, dff=512, dropout_rate=0.1):
        super(TransformerModel, self).__init__()

        # 入力層
        self.input_layer = layers.Dense(d_model)  # 次元変換

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
        self.output_layer = layers.Dense(6, activation="linear")

    def call(self, inputs, training=False):
        x = self.input_layer(inputs)  # 次元変換

        for i in range(len(self.encoder_layers)):
            attn_output = self.encoder_layers[i](x, x, x)
            attn_output = self.dropout_layers[i](attn_output, training=training)
            x = self.norm_layers[i](x + attn_output)

            ffn_output = self.ffn_layers[i](x)
            ffn_output = self.dropout_layers[i](ffn_output, training=training)
            x = self.norm_layers[i](x + ffn_output)

        return self.output_layer(x[:, -1, :])  # 最後のタイムステップの出力を使用

# --- モデルのコンパイル ---
def build_transformer():
    model = TransformerModel()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )
    return model

# --- ここからデータをロードし，データセットを作成する処理 ---
if __name__ == "__main__":
    # クラスをインスタンス化してデータをロード
    data_loader = DataLoader(data_dir="Data_Label/Gym")
    x_data, y_label = data_loader.load_data()

    # DataProcessor をインスタンス化してデータセットを作成
    data_processor = DataProcessor(x_data, y_label, batch_size=32)

    # データセットを取得
    train_dataset, val_dataset, test_dataset = data_processor.get_datasets()

    # モデルを構築
    transformer = build_transformer()

    # モデルの学習
    transformer.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=10,  # エポック数
    )

    # モデルの評価
    test_loss, test_mae = transformer.evaluate(test_dataset)
    print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

    # **予測の実行**
    print("=== モデルによる予測開始 ===")

    # テストデータセットから1つのバッチを取得
    test_iter = iter(test_dataset)
    x_test_sample, y_test_sample = next(test_iter)

    # 予測を実行
    predictions = transformer.predict(x_test_sample)

    # 予測結果を表示
    print("Actual y_test:", y_test_sample.numpy()[:5])  # 最初の5つのラベル
    print("Predicted y:", predictions[:5])  # 最初の5つの予測結果
