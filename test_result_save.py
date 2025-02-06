import numpy as np
import os

class TestResultSaver:
    def __init__(self, save_dir="test_results"):
        """
        テストデータの予測値と真のラベルを保存するクラス。

        Args:
            save_dir (str): 保存ディレクトリのパス（デフォルトは 'test_results'）
        """
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)  # 保存ディレクトリを作成

    def save_results(self, test_dataset, model):
        """
        テストデータの予測値と真のラベルを `.npy` 形式で保存する。

        Args:
            test_dataset (tf.data.Dataset): テストデータセット
            model (tf.keras.Model): 学習済みのLSTMモデル
        """
        print("=== テストデータの予測と保存を開始 ===")

        # ✅ テストデータの予測値と真値を保存するリスト
        test_predictions_list = []
        test_true_values_list = []

        # ✅ テストデータセットをバッチごとに処理
        for test_x_batch, test_y_batch in test_dataset:
            batch_test_predictions = model.predict(test_x_batch)
            test_predictions_list.append(batch_test_predictions)
            test_true_values_list.append(test_y_batch.numpy())

        # ✅ NumPy 配列に変換
        test_predictions = np.concatenate(test_predictions_list, axis=0)
        test_true_values = np.concatenate(test_true_values_list, axis=0)

        # ✅ 予測結果と真値を `.npy` 形式で保存
        np.save(os.path.join(self.save_dir, "test_predictions.npy"), test_predictions)
        np.save(os.path.join(self.save_dir, "test_true_values.npy"), test_true_values)

        print("✅ テストデータの予測値と真値を保存しました！")
        print(f"保存先: {self.save_dir}")
        print(f"test_predictions.npy の形状: {test_predictions.shape}")
        print(f"test_true_values.npy の形状: {test_true_values.shape}")
