# training_summary_saver.py
import os

class TrainingSummarySaver:
    def __init__(self, filename="training_summary.txt"):
        """
        学習情報を保存するクラス
        """
        self.filename = filename

    def save_summary(self, batch_size, final_train_mse, final_val_mse, test_mse, model):
        """
        学習情報（MSEやモデル構造）をTXTファイルに保存
        """
        # ✅ モデルの構造を取得
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        model_summary = "\n".join(model_summary)

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
        with open(self.filename, "w") as f:
            f.write(summary_text)

        print(f"=== 学習情報を '{self.filename}' に保存しました ===")
