import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt

class RSSIInterpolator:
    def __init__(self, grid_size=(100, 100), x_coords=None, y_coords=None):
        """
        RSSIデータ（6要素のnp.array）を100×100の行列にスプライン補間するクラス。

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
        6つのRSSI値を補間し、100×100の行列を生成する。

        Args:
            rssi_values (np.array): 形状 (6,) のRSSI測定値（順番: [00, 01, 02, 10, 11, 12]）

        Returns:
            np.array: 100×100の補間後のRSSIデータ
        """
        if rssi_values.shape != (6,):
            raise ValueError("rssi_values は 6要素の np.array である必要があります。")

        # 6つのRSSI値を (2,3) の形状に変換
        z_values = np.array([
            [rssi_values[0], rssi_values[1], rssi_values[2]],  # 上側の3点
            [rssi_values[3], rssi_values[4], rssi_values[5]]   # 下側の3点
        ])

        # 2D スプライン補間（kx=1, ky=2: x方向1次、y方向2次）
        spline = RectBivariateSpline(self.x_coords, self.y_coords, z_values, kx=1, ky=2)

        # 補間後の行列を生成
        interpolated_grid = spline(self.xi, self.yi)

        return interpolated_grid

    def plot_heatmap(self, rssi_values, cmap="viridis"):
        """
        RSSI補間結果をヒートマップとして可視化する。

        Args:
            rssi_values (np.array): 形状 (6,) のRSSI測定値
            cmap (str): カラーマップ（デフォルト: "viridis"）
        """
        interpolated_grid = self.interpolate(rssi_values)

        plt.figure(figsize=(6, 5))
        plt.imshow(interpolated_grid, extent=(0, 1, 0, 2), origin='lower', aspect='auto', cmap=cmap)
        plt.colorbar(label="RSSI (dB)")
        plt.title("Interpolated RSSI Heatmap")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.show()

# デバッグ用の実行ブロック
if __name__ == "__main__":
    # ✅ テスト用のRSSI値（6要素のnp.array）
    test_rssi_values = np.array([-72.70863, -73.76971, -84.82019, -65.957184, -71.67296, -70.84751])

    # ✅ インスタンス作成
    interpolator = RSSIInterpolator()

    # ✅ 100×100の行列に補間
    interpolated_grid = interpolator.interpolate(test_rssi_values)
    print("補間後の行列の形状:", interpolated_grid.shape)  # (100, 100)

    # ✅ ヒートマップをプロット
    interpolator.plot_heatmap(test_rssi_values)
