import os
import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt

class RSSIInterpolator:
    def __init__(self, grid_size=(100, 100), x_coords=None, y_coords=None):
        """
        RSSIデータ（n×6のnp.array）を100×100の行列にスプライン補間するクラス。

        Args:
            grid_size (tuple): 出力する補間後の行列サイズ（デフォルトは100×100）
            x_coords (np.array): 測定機器のX座標（デフォルト: [0, 1]）
            y_coords (np.array): 測定機器のY座標（デフォルト: [0, 1, 2]）
        """
        self.grid_size = grid_size
        self.x_coords = x_coords if x_coords is not None else np.array([0, 1])
        self.y_coords = y_coords if y_coords is not None else np.array([0, 1, 2])

        # 補間後の座標（100×100に補間）
        self.xi = np.linspace(self.x_coords.min(), self.x_coords.max(), self.grid_size[0])
        self.yi = np.linspace(self.y_coords.min(), self.y_coords.max(), self.grid_size[1])

    def interpolate(self, rssi_values):
        """
        n個のRSSI測定値セットを補間し、100×100の行列を n 回生成する。

        Args:
            rssi_values (np.array): 形状 (n, 6) のRSSI測定値（各行が [00, 01, 02, 10, 11, 12]）

        Returns:
            np.array: 形状 (n, 100, 100) の補間後のRSSIデータ
        """
        if len(rssi_values.shape) != 2 or rssi_values.shape[1] != 6:
            raise ValueError("rssi_values は (n, 6) の np.array である必要があります。")

        n = rssi_values.shape[0]  # サンプル数
        interpolated_grids = np.zeros((n, self.grid_size[0], self.grid_size[1]))  # 結果を格納する配列

        for i in range(n):
            # 6つのRSSI値を (2,3) の形状に変換
            z_values = np.array([
                [rssi_values[i, 0], rssi_values[i, 1], rssi_values[i, 2]],  # 上側の3点
                [rssi_values[i, 3], rssi_values[i, 4], rssi_values[i, 5]]   # 下側の3点
            ])

            # 2D スプライン補間（kx=1, ky=2: x方向1次、y方向2次）
            spline = RectBivariateSpline(self.x_coords, self.y_coords, z_values, kx=1, ky=2)

            # 補間後の行列を保存
            interpolated_grids[i] = spline(self.xi, self.yi)

        return interpolated_grids

    def plot_heatmaps(self, rssi_values, cmap="viridis", num_samples=5):
        """
        複数のRSSI補間結果をヒートマップとして可視化する。

        Args:
            rssi_values (np.array): 形状 (n, 6) のRSSI測定値
            cmap (str): カラーマップ（デフォルト: "viridis"）
            num_samples (int): 表示するヒートマップの数（デフォルト: 5）
        """
        interpolated_grids = self.interpolate(rssi_values)

        # 最大 num_samples 個のヒートマップを表示
        for i in range(min(num_samples, rssi_values.shape[0])):
            plt.figure(figsize=(6, 5))
            plt.imshow(interpolated_grids[i], extent=(0, 1, 0, 2), origin='lower', aspect='auto', cmap=cmap)
            plt.colorbar(label="RSSI (dB)")
            plt.title(f"Interpolated RSSI Heatmap {i+1}")
            plt.xlabel("X Coordinate")
            plt.ylabel("Y Coordinate")
            plt.show()


# デバッグ用の実行ブロック
if __name__ == "__main__":
    # ✅ "test_results/test_predictions.npy" のフルパスを取得
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 現在のスクリプトのディレクトリ
    test_results_path = os.path.join(current_dir, "test_results", "test_predictions.npy")

    # ✅ テストデータをロード
    if not os.path.exists(test_results_path):
        raise FileNotFoundError(f"テストデータ {test_results_path} が見つかりません！")

    test_rssi_values = np.load(test_results_path)

    # ✅ 形状をチェック
    print("ロードしたテストデータの形状:", test_rssi_values.shape)  # (n, 6) を期待

    # ✅ インスタンス作成
    interpolator = RSSIInterpolator()

    # ✅ 100×100の行列に補間（n回）
    interpolated_grids = interpolator.interpolate(test_rssi_values)
    print("補間後の行列の形状:", interpolated_grids.shape)  # (n, 100, 100)

    # ✅ ヒートマップをプロット（最大3つ）
    interpolator.plot_heatmaps(test_rssi_values, num_samples=3)
