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

    def plot_spline_difference_map(self):
        """
        予測値と真値のスプライン補間の差をヒートマップとして可視化。
        """
        # ✅ 保存されたデータをロード
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
