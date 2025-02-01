import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from load_data_label import DataLoader
from create_dataset import DataProcessor

# TensorFlow のデバッグメッセージを抑制
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- LSTM モデルの構築 ---
class LSTMModel(keras.Model):
    def __init__(self, num_units=128, num_layers=3, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.lstm_layers = []
        
        # 最初の LSTM 層
        self.lstm_layers.append(layers.LSTM(num_units, return_sequences=True, dropout=dropout_rate))
        self.lstm_layers.append(layers.BatchNormalization())  # ✅ BatchNormalization を追加

        # 中間の LSTM 層
        for _ in range(num_layers - 2):
            self.lstm_layers.append(layers.LSTM(num_units, return_sequences=True, dropout=dropout_rate))
            self.lstm_layers.append(layers.BatchNormalization())  # ✅ BatchNormalization

        # 最後の LSTM 層
        self.lstm_layers.append(layers.LSTM(num_units, return_sequences=False, dropout=dropout_rate))

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

    # ✅ Optimizer を `AdamW` から `RMSprop` に変更
    optimizer = keras.optimizers.RMSprop(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"]
    )
    return model

# --- メイン処理 ---
if __name__ == "__main__":
    print("=== データの作成を開始 ===")
    data_loader = DataLoader(data_dir="Data_Label/Gym")
    x_data, y_label = data_loader.load_data()

    print("=== データセットの作成を開始 ===")
    data_processor = DataProcessor(x_data, y_label, batch_size=64)
    train_dataset, val_dataset, test_dataset = data_processor.get_datasets()

    sample_input_shape = x_data.shape[1:]  # 例: (10, 6)

    print("=== LSTM モデルの構築 ===")
    lstm_model = build_lstm(sample_input_shape)
    lstm_model.summary()

    print("=== モデルの学習を開始 ===")
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)

    lstm_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=20,
        callbacks=[early_stopping, lr_scheduler]
    )

    print("=== モデルの評価 ===")
    test_loss, test_mae = lstm_model.evaluate(test_dataset)
    print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

    print("=== モデルの保存 ===")
    lstm_model.save("lstm_model", save_format="tf")

    print("=== モデルの予測テスト ===")
    test_iter = iter(test_dataset)
    x_test_sample, y_test_sample = next(test_iter)

    predictions = lstm_model.predict(x_test_sample)

    print("Actual y_test:", y_test_sample.numpy()[:5])
    print("Predicted y:", predictions[:5])
