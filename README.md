###電波地図予測
機械学習による時系列データの処理と予測モデルの実装。

概要
本プロジェクトは、RSSIデータの補間、時系列データの学習、Transformer/LSTM モデルによる予測、損失の記録、予測結果の保存などを行う機能を提供する。

ファイル構成
1. rssi_interpolator.py
RSSIデータ（n×6のNumPy配列）をスプライン補間して100×100の行列を生成するクラス RSSIInterpolator を提供。
関数:
interpolate(rssi_values): RSSI測定値を補間。
plot_heatmaps(rssi_values, title, cmap="viridis", num_samples=5): ヒートマップで可視化。
2. test_result_save.py
TestResultSaver クラスを提供し、テストデータの予測値と真値を .npy 形式で保存。
関数:
save_results(test_dataset, model, y_min, y_max, save_dir="test_results"): テストデータの予測結果を保存。
3. transformer.py
Transformer モデル TimeSeriesTransformer を実装。
位置エンコーディングを使用し、時系列データを学習・予測。
TestResultSaver を使って結果を .npy に保存。
4. lstm_model.py
LSTM モデル LSTMModel を実装し、時系列データを学習・予測。
3層の LSTM (return_sequences=True) を使用し、最終層で 6 次元の線形出力を生成。
TestResultSaver を使って結果を .npy に保存。
LossLogger を使用し、学習の損失を記録。
関数:
build_lstm(input_shape): LSTM モデルの構築。
call(inputs, training=False): LSTM の順伝播を実行。
5. load_data_label.py
DataLoader クラスを提供し、x_data.npy と y_label.npy を読み込む。
NumPy 配列の形状を確認し、存在しない場合はエラーを発生させる。
6. create_dataset.py
DataProcessor クラスを提供し、データを TensorFlow Dataset に変換。
正規化（Min-Max スケーリング or Z-score）を適用し、学習・検証・テスト用に分割。
7. loss_logger.py
LossLogger クラスを提供し、学習・検証・テストの損失を記録。
関数:
on_epoch_end(epoch, logs): 学習/検証損失をログファイルに記録。
save_test_loss(test_loss): テスト損失を記録。
セットアップ
環境構築
必要な Python モジュールをインストール
bash
コピーする
編集する
pip install tensorflow numpy matplotlib scipy
データを配置（例: Data_Label/E420 内に x_data.npy と y_label.npy）
実行方法
Transformer モデルを学習・評価・予測
bash
コピーする
編集する
python transformer.py
LSTM モデル
bash
コピーする
編集する
python lstm_model.py
test_results/ に .npy ファイルとして予測結果が保存される。
ライセンス
MIT License

このままコピーして使ってください。内容の追加や変更があれば教えてください！