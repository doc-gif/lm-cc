import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==========================================
# 1. パスの設定とデータセットの定義
# ==========================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = SCRIPT_DIR.parent.parent

BASE_DIR_PRIOR = PROJECTS_DIR / "lm-cc" / "results"
OUTPUT_BASE_DIR = PROJECTS_DIR / "lm-cc" / "out" / "lm_cc"
LOC_BASE_DIR = PROJECTS_DIR / "lm-cc" / "out" / "loc"

# プロット画像の保存先
PLOT_OUTPUT_DIR = PROJECTS_DIR / "lm-cc" / "out" / "plots"
PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    {
        "name": "HumanEval",
        "lm_cc_json": OUTPUT_BASE_DIR / "humaneval-ier_entropy" / "lm_cc_entropies-CodeLlama-7b-hf-1.0.json",
        "score_json": BASE_DIR_PRIOR / "humaneval-ier" / "results_score.json",
        "loc_json": LOC_BASE_DIR / "humaneval-ier" / "loc.json",
        "score_format": "simple"
    },
    {
        "name": "XcodeEval (APR)",
        "lm_cc_json": OUTPUT_BASE_DIR / "xcodeeval_apr_entropy" / "lm_cc_entropies-CodeLlama-7b-hf-1.0.json",
        "score_json": BASE_DIR_PRIOR / "xcodeeval" / "apr" / "python_test_filtered_results.json",
        "loc_json": LOC_BASE_DIR / "xcodeeval_apr" / "loc.json",
        "score_format": "nested"
    },
    {
        "name": "XcodeEval (Code Translation)",
        "lm_cc_json": OUTPUT_BASE_DIR / "xcodeeval_code_translation_entropy" / "lm_cc_entropies-CodeLlama-7b-hf-1.0.json",
        "score_json": BASE_DIR_PRIOR / "xcodeeval" / "code_translation" / "python2c_test_filtered_results.json",
        "loc_json": LOC_BASE_DIR / "xcodeeval_code_translation" / "loc.json",
        "score_format": "nested"
    }
]

# ==========================================
# 2. utils/correlation.py モジュールのインポート
# ==========================================
if str(PROJECTS_DIR / "lm-cc") not in sys.path:
    sys.path.append(str(PROJECTS_DIR / "lm-cc"))

try:
    from scripts.utils.correlation import get_grouped_partial_corr, group_by_metric
except ImportError as e:
    print(f"❌ [致命的なエラー] utils/correlation.py のインポートに失敗しました。\n詳細: {e}")
    sys.exit(1)


# ==========================================
# 3. データ読み込み関数群
# ==========================================
def load_lmcc(path):
    if not path.exists(): return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["task_id"]: item["lm_cc"] for item in data}


def load_scores(path, format_type):
    if not path.exists(): return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    scores = {}
    if format_type == "simple":
        for k, v in data.items(): scores[k.split("__")[0]] = v
    elif format_type == "nested":
        for k, v in data.items(): scores[k] = v.get("pass@1", 0.0)
    return scores


def load_loc(path):
    if not path.exists(): return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ==========================================
# 4. プロット作成処理
# ==========================================
def plot_scatter(dataset_name, all_metrics, all_scores, mean_metrics, mean_scores, r_val, p_val):
    # 下部の余白を確保するため、縦幅を少し大きめに設定
    plt.figure(figsize=(9, 7))
    sns.set_theme(style="whitegrid")

    # 1. 全個別データポイントのプロット
    plt.scatter(
        all_metrics, all_scores,
        color='gray', alpha=0.3, s=20, edgecolor='none',
        label='All Individual Tasks'
    )

    # 2. ビニングされた代表点と回帰線 (トレンドライン) の描画
    sns.regplot(
        x=mean_metrics,
        y=mean_scores,
        ci=None,
        scatter_kws={'s': 100, 'alpha': 0.9, 'edgecolor': 'w', 'color': '#2b8cbe'},
        line_kws={'color': '#f03b20', 'linestyle': '--', 'linewidth': 2},
        label='Binned Representatives'
    )

    # タイトルと軸ラベルの設定
    plt.title(f"LM-CC vs Accuracy (Pass@1)\n{dataset_name}", fontsize=14, pad=15)
    plt.xlabel("LM-CC Score", fontsize=12)
    plt.ylabel("Accuracy (Pass@1)", fontsize=12)

    # y軸（正答率）の範囲を0.0〜1.05に固定
    plt.ylim(-0.05, 1.05)

    # 💡 凡例をグラフの下（X軸ラベルのさらに下）の中央に2列で配置
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True, fontsize=11)

    # 💡 相関係数とp値のテキストを1行にまとめ、凡例のさらに下に配置
    textstr = f"Partial Spearman $r$ = {r_val:.3f}   |   $p$-value "
    if p_val < 0.001:
        textstr += "< 0.001"
    else:
        textstr += f"= {p_val:.3f}"

    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
    plt.gca().text(0.5, -0.28, textstr, transform=plt.gca().transAxes, fontsize=12,
                   verticalalignment='top', horizontalalignment='center', bbox=props)

    plt.tight_layout()

    # 画像保存時に枠外の要素が見切れないように bbox_inches='tight' を指定
    safe_name = dataset_name.replace(" ", "_").replace("(", "").replace(")", "")
    out_path = PLOT_OUTPUT_DIR / f"scatter_partial_{safe_name}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  📸 プロットを保存しました: {out_path}")


def process_dataset(dataset_info):
    ds_name = dataset_info["name"]
    print(f"\n{'=' * 60}")
    print(f"🚀 プロット作成開始: {ds_name}")
    print(f"{'=' * 60}")

    lmcc_data = load_lmcc(dataset_info["lm_cc_json"])
    scores_data = load_scores(dataset_info["score_json"], dataset_info["score_format"])
    loc_data = load_loc(dataset_info["loc_json"])

    if not lmcc_data or not scores_data or not loc_data:
        print("  ❌ 必要なデータが揃っていないため、スキップします。")
        return

    all_score = []
    all_lmcc = []
    all_loc = []

    valid_count = 0
    for task_id, lmcc_val in lmcc_data.items():
        if task_id in scores_data and task_id in loc_data:
            all_score.append(scores_data[task_id])
            all_lmcc.append(lmcc_val)
            all_loc.append(loc_data[task_id])
            valid_count += 1

    if valid_count < 10:
        print("  ⚠️ データ件数が少なすぎるためスキップします。")
        return

    all_score = np.array(all_score)
    all_lmcc = np.array(all_lmcc)
    all_loc = np.array(all_loc)

    # 1. 偏相関（LOCを考慮）の最適なグループ化パラメータを取得
    try:
        res_partial, best_min_cnt = get_grouped_partial_corr(all_score, all_lmcc, all_loc)
    except Exception as e:
        print(f"  ❌ 相関計算エラー: {e}")
        return

    if res_partial and "partial_correlation" in res_partial and best_min_cnt is not None:
        r_partial = res_partial["partial_correlation"]["spearman-r"]
        p_partial = res_partial["partial_correlation"]["spearman-pval"]

        print(f"  🔍 最適なBinサイズ (min_cnt): {best_min_cnt}")

        # 2. 代表点（平均と中央値）を取得
        mean_scores, _, mean_metrics, _ = group_by_metric(all_score, all_lmcc, None, min_cnt=best_min_cnt)

        # 3. グラフの描画
        plot_scatter(ds_name, all_lmcc, all_score, mean_metrics, mean_scores, r_partial, p_partial)
    else:
        print(f"  ❌ 相関係数が有効でないため、プロットをスキップします。")


def main():
    for dataset_info in DATASETS:
        process_dataset(dataset_info)
    print("\n🎉 全てのプロット作成が完了しました！")


if __name__ == "__main__":
    main()