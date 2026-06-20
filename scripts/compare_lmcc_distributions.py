import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = SCRIPT_DIR.parent.parent
OUTPUT_BASE_DIR = PROJECTS_DIR / "lm-cc" / "out" / "lm_cc"

PAIRS = [
    {
        "task": "HumanEval",
        "original": OUTPUT_BASE_DIR / "humaneval-ier_entropy" / "lm_cc_entropies-CodeLlama-7b-hf-1.0.json",
        "simplified": OUTPUT_BASE_DIR / "humaneval-ier-simplified_entropy" / "lm_cc_entropies-CodeLlama-7b-hf-1.0.json"
    },
    {
        "task": "XcodeEval (APR)",
        "original": OUTPUT_BASE_DIR / "xcodeeval_apr_entropy" / "lm_cc_entropies-CodeLlama-7b-hf-1.0.json",
        "simplified": OUTPUT_BASE_DIR / "xcodeeval_apr-simplified_entropy" / "lm_cc_entropies-CodeLlama-7b-hf-1.0.json"
    },
    {
        "task": "XcodeEval (Code Translation)",
        "original": OUTPUT_BASE_DIR / "xcodeeval_code_translation_entropy" / "lm_cc_entropies-CodeLlama-7b-hf-1.0.json",
        "simplified": OUTPUT_BASE_DIR / "xcodeeval_code_translation-simplified_entropy" / "lm_cc_entropies-CodeLlama-7b-hf-1.0.json"
    }
]


def load_lmcc_dict(path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["task_id"]: item["lm_cc"] for item in data}


def main():
    plot_data = []

    for pair in PAIRS:
        task = pair["task"]
        orig_dict = load_lmcc_dict(pair["original"])
        simp_dict = load_lmcc_dict(pair["simplified"])

        # 💡 両方に存在するタスクID (ペア) だけを抽出
        common_tasks = set(orig_dict.keys()).intersection(set(simp_dict.keys()))

        if not common_tasks:
            print(f"⚠️ {task} のペアデータが見つかりません。")
            continue

        orig_vals = []
        simp_vals = []
        diffs = []

        for t_id in common_tasks:
            o_val = orig_dict[t_id]
            s_val = simp_dict[t_id]
            orig_vals.append(o_val)
            simp_vals.append(s_val)
            diffs.append(o_val - s_val)  # 正ならLM-CCが下がった(改善)、負なら上がった(悪化)

            plot_data.append({"Task": task, "Version": "Original", "LM_CC": o_val, "Task_ID": t_id})
            plot_data.append({"Task": task, "Version": "Simplified", "LM_CC": s_val, "Task_ID": t_id})

        # ウィルコクソンの符号順位検定 (対応のあるデータ向け)
        try:
            stat, p_val = stats.wilcoxon(orig_vals, simp_vals, alternative='greater')
            mean_diff = sum(diffs) / len(diffs)

            print(f"\n🚀 {task} のペアワイズ比較 (N={len(common_tasks)}ペア):")
            print(f"  - Original 平均: {sum(orig_vals) / len(orig_vals):.2f}")
            print(f"  - Simplified 平均: {sum(simp_vals) / len(simp_vals):.2f}")
            print(f"  - 平均変化量: {mean_diff:+.2f} " + ("(下がった📉)" if mean_diff > 0 else "(上がった📈)"))
            print(f"  - ウィルコクソン p値: {p_val:.4e} " + ("(有意に低下 ✅)" if p_val < 0.05 else "(低下せず ❌)"))
        except Exception as e:
            print(f"  ❌ {task} の検定エラー: {e}")

    # ペア変化を可視化するグラフ (Before/After)
    if plot_data:
        df = pd.DataFrame(plot_data)

        # HumanEval だけ抽出して Before-After の傾きを見るプロット
        he_df = df[df["Task"] == "HumanEval"]
        if not he_df.empty:
            plt.figure(figsize=(8, 6))
            sns.lineplot(data=he_df, x="Version", y="LM_CC", units="Task_ID", estimator=None, color="gray", alpha=0.3)
            sns.pointplot(data=he_df, x="Version", y="LM_CC", color="blue", errorbar=None, markers="o")
            plt.title("LM-CC Change per Task: HumanEval (Original vs Simplified)")
            plt.ylabel("LM-CC Score")

            out_img = PROJECTS_DIR / "lm-cc" / "out" / "lmcc_paired_comparison.png"
            plt.savefig(out_img, dpi=300)
            print(f"\n🎉 グラフ画像を保存しました: {out_img}")


if __name__ == "__main__":
    main()