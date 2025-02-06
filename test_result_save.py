import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.interpolate as interp

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

    def plot_spline_difference_map(self):
        """
        予測値と真値のスプライン補間の差をヒートマップとして可視化。
        """
        # ✅ 保存されたデータを読み込む
        pred_path = os.path.join(self.save_dir, "test_predictions.npy")
        true_path = os.path.join(self.save_dir, "test_true_values.npy")

        if not os.path.exists(pred_path) or not os.path.exists(true_path):
            print("❌ 保存された予測データまたは真値データが見つかりません。")
            return

        test_predictions = np.load(pred_path)
        test_true_values = np.load(true_path)

        num_samples, num_features = test_predictions.shape
        x = np.linspace(0, 1, num_samples)  # 正規化されたx軸

        # ✅ スプライン補間
        pred_splines = [interp.InterpolatedUnivariateSpline(x, test_predictions[:, i]) for i in range(num_features)]
        true_splines = [interp.InterpolatedUnivariateSpline(x, test_true_values[:, i]) for i in range(num_features)]

        # ✅ 差分マップを作成
        x_fine = np.linspace(0, 1, 100)  # 高解像度で補間
        diff_map = np.array([pred_splines[i](x_fine) - true_splines[i](x_fine) for i in range(num_features)])

        # ✅ ヒートマップを描画
        plt.figure(figsize=(10, 6))
        plt.imshow(diff_map, aspect='auto', cmap='coolwarm', extent=[0, 1, 0, num_features])
        plt.colorbar(label="Prediction - True")
        plt.xlabel("Normalized Sample Index")
        plt.ylabel("Feature Index")
        plt.title("Spline Interpolation Difference Map")
        plt.show()

        print("✅ スプライン差分ヒートマップを表示しました！")
