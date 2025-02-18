import os

class TrainingSummarySaver:
    def __init__(self, filename="training_summary.txt"):
        """
        学習情報を保存するクラス
        """
        self.filename = filename

    def save_summary(self, batch_size, final_train_mse, final_val_mse, test_mse, model, save_path="."):
        """
        学習情報（MSEやモデル構造）をTXTファイルに保存
        Args:
            batch_size (int): バッチサイズ
            final_train_mse (float): 最終エポックの学習 MSE
            final_val_mse (float): 最終エポックの検証 MSE
            test_mse (float): テストデータの MSE
            model (keras.Model): 保存するモデル
            save_path (str): 保存先のディレクトリパス（デフォルトはカレントディレクトリ）
        """
        # ✅ モデルの構造を取得
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        model_summary = "\n".join(model_summary)

        # ✅ 保存ディレクトリを作成（存在しない場合）
        os.makedirs(save_path, exist_ok=True)

        # ✅ 保存するファイルのフルパス
        save_file = os.path.join(save_path, self.filename)

        # ✅ 学習情報をフォーマット
        summary_text = f"""Training Summary:
Batch Size: {batch_size}
Final Training MSE: {final_train_mse:.6f}
Final Validation MSE: {final_val_mse:.6f}
Test MSE: {test_mse:.6f}

Model Structure:
{model_summary}
"""

        # ✅ 学習情報をTXTに保存
        with open(save_file, "w") as f:
            f.write(summary_text)

        print(f"=== 学習情報を '{save_file}' に保存しました ===")
