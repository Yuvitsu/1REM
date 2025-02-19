# main.py
import os
import tensorflow as tf
from tensorflow import keras
from load_data_label import DataLoader
from create_dataset import DataProcessor
from loss_logger import LossLogger
from test_result_save import TestResultSaver
import numpy as np
from training_summary_saver import TrainingSummarySaver  # ✅ クラスをインポート

# ✅ LossLogger のインスタンスを作成
loss_logger = LossLogger(model_name="deep_transformer_model")

# ✅ 学習情報保存クラスのインスタンス
summary_saver = TrainingSummarySaver()

# ✅ データのロード
data_loader = DataLoader(data_dir="Data_Label/Gym")
x_data, y_label = data_loader.load_data()

# ✅ バッチサイズ定義
batch_size = 32

# ✅ データの正規化
data_processor = DataProcessor(x_data, y_label, batch_size=batch_size, normalization_method="minmax")
train_dataset, val_dataset, test_dataset = data_processor.get_datasets()

# ✅ モデルの構築
model = keras.models.load_model("deep_transformer_model")  # 事前に学習済みモデルを読み込む

# ✅ モデルの学習
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=[loss_logger]
)

# ✅ 最終エポックの MSE を取得
final_train_mse = history.history['mse'][-1]
final_val_mse = history.history['val_mse'][-1]

# ✅ モデルの評価
test_loss, test_mse = model.evaluate(test_dataset)

# ✅ 学習情報を保存
summary_saver.save_summary(batch_size, final_train_mse, final_val_mse, test_mse, model,)
