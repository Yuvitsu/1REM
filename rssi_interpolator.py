import os
import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt

class RSSIInterpolator:
    def __init__(self, grid_size=(100, 100), x_coords=None, y_coords=None):
        """
        RSSIデータ（n×6のnp.array）を100×100の行列にスプライン補間するクラス。
        """
        self.grid_size = grid_size
        self.x_coords = x_coords if x_coords is not None else np.array([0, 1])
        self.y_coords = y_coords if y_coords is not None else np.array([0, 1, 2])
        self.xi = np.linspace(self.x_coords.min(), self.x_coords.max(), self.grid_size[0])
        self.yi = np.linspace(self.y_coords.min(), self.y_coords.max(), self.grid_size[1])

    def interpolate(self, rssi_values):
        """
        RSSI測定値セットを補間し、100×100の行列を生成する。
        """
        if len(rssi_values.shape) != 2 or rssi_values.shape[1] != 6:
            raise ValueError("rssi_values は (n, 6) の np.array である必要があります。")

        n = rssi_values.shape[0]
        interpolated_grids = np.zeros((n, self.grid_size[0], self.grid_size[1]))

        for i in range(n):
            z_values = np.array([
                [rssi_values[i, 0], rssi_values[i, 1], rssi_values[i, 2]],
                [rssi_values[i, 3], rssi_values[i, 4], rssi_values[i, 5]]
            ])
            spline = RectBivariateSpline(self.x_coords, self.y_coords, z_values, kx=1, ky=2)
            interpolated_grids[i] = spline(self.xi, self.yi)

        return interpolated_grids

    def plot_heatmaps(self, rssi_values, title, cmap="viridis", num_samples=5):
        """
        RSSI補間結果のヒートマップを可視化。
        """
        interpolated_grids = self.interpolate(rssi_values)
        for i in range(min(num_samples, rssi_values.shape[0])):
            plt.figure(figsize=(6, 5))
            plt.imshow(interpolated_grids[i], extent=(0, 1, 0, 2), origin='lower', aspect='auto', cmap=cmap)
            plt.colorbar(label="RSSI (dB)")
            plt.title(f"{title} {i+1}")
            plt.xlabel("X Coordinate")
            plt.ylabel("Y Coordinate")
            plt.show()

    def compute_error_map(self, pred_values, true_values):
        """
        予測値と真値の補間後の誤差のヒートマップを作成。
        """
        pred_interpolated = self.interpolate(pred_values)
        true_interpolated = self.interpolate(true_values)
        error_maps = np.abs(pred_interpolated - true_interpolated)
        return error_maps

    def compute_mse(self, pred_values, true_values):
        """
        予測と真値の MSE を計算。
        """
        pred_interpolated = self.interpolate(pred_values)
        true_interpolated = self.interpolate(true_values)
        mse = np.mean((pred_interpolated - true_interpolated) ** 2)
        return mse

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pred_path = os.path.join(current_dir, "test_results", "test_predictions.npy")
    true_path = os.path.join(current_dir, "test_results", "test_true_values.npy")

    if not os.path.exists(pred_path) or not os.path.exists(true_path):
        raise FileNotFoundError("予測データまたは真値データが見つかりません！")

    pred_rssi_values = np.load(pred_path)
    true_rssi_values = np.load(true_path)
    print("ロードしたデータの形状:", pred_rssi_values.shape, true_rssi_values.shape)

    interpolator = RSSIInterpolator()
    interpolator.plot_heatmaps(true_rssi_values, "True RSSI Heatmap", num_samples=3)
    interpolator.plot_heatmaps(pred_rssi_values, "Predicted RSSI Heatmap", num_samples=3)
    
    error_maps = interpolator.compute_error_map(pred_rssi_values, true_rssi_values)
    interpolator.plot_heatmaps(error_maps, "Error Heatmap", cmap="coolwarm", num_samples=3)
    
    mse = interpolator.compute_mse(pred_rssi_values, true_rssi_values)
    print(f"Mean Squared Error (MSE) between predictions and true values: {mse}")
