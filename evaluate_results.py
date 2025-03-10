import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

class ModelResultEvaluator:
    def __init__(self, dataset_name, model_name, predictions_path, true_values_path, output_dir_base="test_results"):
        """
        モデルの予測結果を評価するクラス。

        Args:
            dataset_name (str): データセット名 (例: "E420", "Gym")
            model_name (str): モデル名 (例: "LSTM", "Transformer")
            predictions_path (str): 予測値 (`test_predictions.npy`) のパス
            true_values_path (str): 真値 (`test_true_values.npy`) のパス
            output_dir_base (str): 結果を保存するディレクトリのベース
        """
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.predictions_path = predictions_path
        self.true_values_path = true_values_path
        self.output_dir = os.path.join(output_dir_base, dataset_name, f"{model_name}_heatmaps")
        os.makedirs(self.output_dir, exist_ok=True)

        # スプライン補間の設定
        self.grid_size = (100, 100)
        self.x_coords = np.array([0, 1])
        self.y_coords = np.array([0, 1, 2])
        self.xi = np.linspace(self.x_coords.min(), self.x_coords.max(), self.grid_size[0])
        self.yi = np.linspace(self.y_coords.min(), self.y_coords.max(), self.grid_size[1])

        # データのロード
        self.test_predictions = None
        self.test_true_values = None
        self.test_predictions_interp = None
        self.test_true_values_interp = None
        self.diff_array = None
        self.mse_value = None

    def load_data(self):
        """NumPy ファイルをロード"""
        print(f"=== {self.dataset_name} - {self.model_name}: データのロード ===")
        self.test_predictions = np.load(self.predictions_path)  # (サンプル数, 6)
        self.test_true_values = np.load(self.true_values_path)  # (サンプル数, 6)
        print(f"✅ {self.dataset_name} - {self.model_name} test_predictions.shape: {self.test_predictions.shape}")
        print(f"✅ {self.dataset_name} - {self.model_name} test_true_values.shape: {self.test_true_values.shape}")

    def interpolate(self, data):
        """
        スプライン補間を適用し、(サンプル数, 100, 100) のデータを生成。
        """
        n_samples = data.shape[0]
        interpolated_grids = np.zeros((n_samples, self.grid_size[0], self.grid_size[1]))

        for i in range(n_samples):
            z_values = np.array([
                [data[i, 0], data[i, 1], data[i, 2]],
                [data[i, 3], data[i, 4], data[i, 5]]
            ])
            spline = RectBivariateSpline(self.x_coords, self.y_coords, z_values, kx=1, ky=2)
            interpolated_grids[i] = spline(self.xi, self.yi)

        return interpolated_grids

    def apply_interpolation(self):
        """スプライン補間を適用"""
        print(f"=== {self.dataset_name} - {self.model_name}: スプライン補間を適用 ===")
        self.test_predictions_interp = self.interpolate(self.test_predictions)
        self.test_true_values_interp = self.interpolate(self.test_true_values)

    def compute_difference(self):
        """補間後のデータの差分を計算"""
        print(f"=== {self.dataset_name} - {self.model_name}: 差分を計算 ===")
        self.diff_array = self.test_predictions_interp - self.test_true_values_interp

    def compute_mse(self):
        """MSE を計算"""
        print(f"=== {self.dataset_name} - {self.model_name}: MSE を計算 ===")
        self.mse_value = np.mean(self.diff_array ** 2)
        mse_path = os.path.join(self.output_dir, "mse_result.txt")
        with open(mse_path, "w") as f:
            f.write(f"Mean Squared Error (MSE): {self.mse_value}\n")
        print(f"✅ {self.dataset_name} - {self.model_name} MSE 計算完了: {self.mse_value}")

    def save_heatmaps(self, data, title_prefix, filename_prefix):
        """ヒートマップを保存"""
        for i in range(min(5, len(data))):  # 最初の5枚を保存
            plt.figure(figsize=(6, 6))
            plt.imshow(data[i], cmap="viridis", aspect="auto")
            plt.colorbar()
            plt.title(f"{self.dataset_name} - {self.model_name}: {title_prefix} {i+1}")
            plt.savefig(os.path.join(self.output_dir, f"{filename_prefix}_{i+1}.png"))
            plt.close()
        print(f"✅ {self.dataset_name} - {self.model_name}: {title_prefix} のヒートマップを保存しました。")

    def save_all_heatmaps(self):
        """補間後の予測値、真値、差分のヒートマップを保存"""
        print(f"=== {self.dataset_name} - {self.model_name}: ヒートマップを保存 ===")
        self.save_heatmaps(self.test_predictions_interp, "Test Predictions Interpolated", "test_predictions_interp")
        self.save_heatmaps(self.test_true_values_interp, "Test True Values Interpolated", "test_true_values_interp")
        self.save_heatmaps(self.diff_array, "Difference (Prediction - True)", "diff_heatmap")

    def evaluate(self):
        """評価を一括実行"""
        self.load_data()
        self.apply_interpolation()
        self.compute_difference()
        self.compute_mse()
        self.save_all_heatmaps()
        print(f"✅ {self.dataset_name} - {self.model_name}: ヒートマップと MSE の結果を {self.output_dir} に保存しました。")

# --- メイン処理 ---
if __name__ == "__main__":
    datasets = ["Gym"]
    models = ["LSTM", "Transformer"]

    for dataset in datasets:
        for model in models:
            evaluator = ModelResultEvaluator(
                dataset_name=dataset,
                model_name=model,
                predictions_path=f"test_results/{model}_results/test_predictions.npy",
                true_values_path=f"test_results/{model}_results/test_true_values.npy"
            )
            evaluator.evaluate()
