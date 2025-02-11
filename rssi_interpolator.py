import os
import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt

class RSSIInterpolator:
    def __init__(self, grid_size=(100, 100), x_coords=None, y_coords=None):
        """
        RSSIデータ（n×m×6のnp.array）を最後の次元(6)だけ100×100の行列にスプライン補間するクラス。
        """
        self.grid_size = grid_size
        self.x_coords = x_coords if x_coords is not None else np.array([0, 1])
        self.y_coords = y_coords if y_coords is not None else np.array([0, 1, 2])
        self.xi = np.linspace(self.x_coords.min(), self.x_coords.max(), self.grid_size[0])
        self.yi = np.linspace(self.y_coords.min(), self.y_coords.max(), self.grid_size[1])
    
    def interpolate(self, rssi_values):
        """
        RSSI測定値セットを最後の次元だけ補間し、(n, m, 100, 100)の行列を生成する。
        """
        if len(rssi_values.shape) != 3 or rssi_values.shape[2] != 6:
            raise ValueError("rssi_values は (n, m, 6) の np.array である必要があります。")
        
        n, m, _ = rssi_values.shape
        interpolated_grids = np.zeros((n, m, self.grid_size[0], self.grid_size[1]))
        
        for i in range(n):
            for j in range(m):
                z_values = np.array([
                    [rssi_values[i, j, 0], rssi_values[i, j, 1], rssi_values[i, j, 2]],
                    [rssi_values[i, j, 3], rssi_values[i, j, 4], rssi_values[i, j, 5]]
                ])
                spline = RectBivariateSpline(self.x_coords, self.y_coords, z_values, kx=1, ky=2)
                interpolated_grids[i, j] = spline(self.xi, self.yi)
        
        return interpolated_grids

if __name__ == "__main__":
    # ダミーデータの作成
    sample_rssi_values = np.random.rand(5, 4, 6) * -50  # (n=5, m=4, 6つのRSSI値)
    interpolator = RSSIInterpolator()
    interpolated_data = interpolator.interpolate(sample_rssi_values)
    print("補間後のデータ形状:", interpolated_data.shape)  # (5, 4, 100, 100)
