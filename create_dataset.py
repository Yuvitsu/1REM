import numpy as np

# NumPy ファイルを読み込む
x_data = np.load('Data_Label/x_data.npy')
y_label = np.load('Data_Label/y_label.npy')

print("x_data shape:", x_data.shape)
print("y_label shape:", y_label.shape)
