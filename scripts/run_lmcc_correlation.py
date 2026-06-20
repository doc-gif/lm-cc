import json
import sys
import numpy as np
from pathlib import Path

# ==========================================
# 1. パスの設定とデータセットの定義 (ユーザー指定)
# ==========================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = SCRIPT_DIR.parent.parent

BASE_DIR_PRIOR = PROJECTS_DIR / "lm-cc" / "results"
OUTPUT_BASE_DIR = PROJECTS_DIR / "lm-cc" / "out" / "lm_cc"
LOC_BASE_DIR = PROJECTS_DIR / "lm-cc" / "out" / "loc"

DATASETS = [
    {
        "name": "HumanEval",
        "lm_cc_json": OUTPUT_BASE_DIR / "humaneval-ier_entropy" / "lm_cc_entropies-CodeLlama-7b-hf-1.0.json",
        "score_json": BASE_DIR_PRIOR / "humaneval-ier" / "results_score.json",
        "loc_json": LOC_BASE_DIR / "humaneval-ier" / "loc.json",
        "score_format": "simple"
    },
    # {
    #     "name": "HumanEval-simplified",
    #     "lm_cc_json": OUTPUT_BASE_DIR / "humaneval-ier-simplified_entropy" / "lm_cc_entropies-CodeLlama-7b-hf-1.0.json",
    #     "score_json": BASE_DIR_PRIOR / "humaneval-ier-simplified" / "results_score_simplified.json",
    #     "loc_json": LOC_BASE_DIR / "humaneval-ier-simplified-top60" / "loc.json",
    #     "score_format": "simple"
    # },
    {
        "name": "XcodeEval(APR)",
        "lm_cc_json": OUTPUT_BASE_DIR / "xcodeeval_apr_entropy" / "lm_cc_entropies-CodeLlama-7b-hf-1.0.json",
        "score_json": BASE_DIR_PRIOR / "xcodeeval" / "apr" / "python_test_filtered_results.json",
        "loc_json": LOC_BASE_DIR / "xcodeeval_apr" / "loc.json",
        "score_format": "nested"
    },
    # {
    #     "name": "XcodeEval-simplified(APR)",
    #     "lm_cc_json": OUTPUT_BASE_DIR / "xcodeeval_apr-simplified_entropy" / "lm_cc_entropies-CodeLlama-7b-hf-1.0.json",
    #     "score_json": BASE_DIR_PRIOR / "xcodeeval" / "apr-simplified" / "python_test_filtered_results.json",
    #     "loc_json": LOC_BASE_DIR / "xcodeeval_apr-simplified-top50" / "loc.json",
    #     "score_format": "nested"
    # },
    {
        "name": "XcodeEval(Code Translation)",
        "lm_cc_json": OUTPUT_BASE_DIR / "xcodeeval_code_translation_entropy" / "lm_cc_entropies-CodeLlama-7b-hf-1.0.json",
        "score_json": BASE_DIR_PRIOR / "xcodeeval" / "code_translation" / "python2c_test_filtered_results.json",
        "loc_json": LOC_BASE_DIR / "xcodeeval_code_translation" / "loc.json",  # 必要に応じて変更
        "score_format": "nested"
    },
    # {
    #     "name": "XcodeEval-simplified(Code Translation)",
    #     "lm_cc_json": OUTPUT_BASE_DIR / "xcodeeval_code_translation-simplified_entropy" / "lm_cc_entropies-CodeLlama-7b-hf-1.0.json",
    #     "score_json": BASE_DIR_PRIOR / "xcodeeval" / "code_translation-simplified" / "python2c_test_filtered_results.json",
    #     "loc_json": LOC_BASE_DIR / "xcodeeval_translation-simplified-top50" / "loc.json",  # 必要に応じて変更
    #     "score_format": "nested"
    # }
]

# ==========================================
# 2. utils/correlation.py モジュールのインポート (ユーザー指定)
# ==========================================
# `scripts.utils.correlation` をインポート可能にするため、lm-ccディレクトリ(PROJECTS_DIR / "lm-cc")をパスに追加
if str(PROJECTS_DIR / "lm-cc") not in sys.path:
    sys.path.append(str(PROJECTS_DIR / "lm-cc"))

try:
    from scripts.utils.correlation import get_grouped_partial_corr, print_metric_result
except ImportError as e:
    print(f"❌ [致命的なエラー] utils/correlation.py のインポートに失敗しました。")
    print(f"詳細: {e}")
    sys.exit(1)


# ==========================================
# 3. データ読み込み関数群
# ==========================================
def load_lmcc(path):
    if not path.exists():
        print(f"    ⚠️ [警告] LM-CCファイルが見つかりません: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["task_id"]: item["lm_cc"] for item in data}


def load_scores(path, format_type):
    if not path.exists():
        print(f"    ⚠️ [警告] スコアファイルが見つかりません: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    scores = {}
    if format_type == "simple":
        for k, v in data.items():
            scores[k.split("__")[0]] = v
    elif format_type == "nested":
        for k, v in data.items():
            scores[k] = v.get("pass@1", 0.0)
    return scores


def load_loc(path):
    if not path.exists():
        print(f"    ⚠️ [警告] LOCファイルが見つかりません: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ==========================================
# 4. 分析結果フォーマット・出力ロジック
# ==========================================
def format_p_value(p):
    """論文フォーマットに合わせて p値を整形する"""
    if p < 0.001:
        return "< 0.001"
    else:
        return f"= {p:.3f}"


def diagnose_none_reason(score, metric, loc=None):
    """相関が計算できず None になった理由を推測する"""
    n = len(score)
    if n < 5:
        return f"データ件数が少なすぎます (N={n})"

    if len(np.unique(score)) <= 1:
        return "正答率(Score)の分散がゼロです（すべて同じ値）"

    if len(np.unique(metric)) <= 1:
        return "LM-CCの分散がゼロです（すべて同じ値）"

    if loc is not None and len(np.unique(loc)) <= 1:
        return "コード行数(LOC)の分散がゼロです（すべて同じ値）"

    return "データグループ化の過程で比較可能なペアが消失した、またはライブラリの計算エラーです"


def analyze_and_print(all_score, all_lmcc, all_loc):
    all_score = np.array(all_score)
    all_lmcc = np.array(all_lmcc)
    all_loc = np.array(all_loc)

    print("\n  📊 [分析結果: LM-CC と LLM正答率の相関]")

    # 1. Zero (偏相関ではない単純な相関)
    try:
        res_zero, _ = get_grouped_partial_corr(all_score, all_lmcc, None)
    except Exception:
        res_zero = None

    if res_zero and "partial_correlation" in res_zero:
        r_zero = res_zero["partial_correlation"]["spearman-r"]
        p_zero = res_zero["partial_correlation"]["spearman-pval"]
        bins = res_zero.get("valid_groups", "N/A")
        print(f"    [Zero]    r = {r_zero:.3f} (p {format_p_value(p_zero)}) | Bins: {bins:>2}")
    else:
        reason = diagnose_none_reason(all_score, all_lmcc)
        print(f"    [Zero]    計算不可 ❌ (理由: {reason})")

    # 2. Partial (LOCを統制変数とした偏相関)
    try:
        res_partial, _ = get_grouped_partial_corr(all_score, all_lmcc, all_loc)
    except Exception:
        res_partial = None

    if res_partial and "partial_correlation" in res_partial:
        r_partial = res_partial["partial_correlation"]["spearman-r"]
        p_partial = res_partial["partial_correlation"]["spearman-pval"]
        bins = res_partial.get("valid_groups", "N/A")
        print(f"    [Partial] r = {r_partial:.3f} (p {format_p_value(p_partial)}) | Bins: {bins:>2}")
    else:
        reason = diagnose_none_reason(all_score, all_lmcc, all_loc)
        print(f"    [Partial] 計算不可 ❌ (理由: {reason})")


# ==========================================
# 5. メイン処理ロジック
# ==========================================
def process_dataset(dataset_info):
    ds_name = dataset_info["name"]
    print(f"\n{'=' * 60}")
    print(f"🚀 相関分析開始: {ds_name}")
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

    print(f"  ✅ 有効データ件数: {valid_count} 件 (突合成功)")

    if valid_count < 10:
        print("  ⚠️ データ件数が少なすぎるため相関分析をスキップします。")
        return

    # 💡 ここからデバッグ用のコードを追加
    from scripts.utils.correlation import group_by_metric

    print("\n  🔍 [Debug] グループ化 (Binning) の内部状態をチェックします...")
    # min_cnt を 10, 20, 30 と変えてみて、グループ数がどうなるか実験
    for test_min_cnt in [10, 20, 30]:
        try:
            # group_by_metric を直接叩いてみる
            mean_scores, mean_metrics, mean_locs, counts = group_by_metric(
                all_score, all_lmcc, all_loc, min_cnt=test_min_cnt
            )
            print(f"    - min_cnt={test_min_cnt} の場合: 圧縮後のグループ数 = {len(mean_scores)} 個")
            if len(mean_scores) < 5:
                print(f"      ⚠️ グループ数が少なすぎます！これでは偏相関は計算できません。")
        except Exception as e:
            print(f"    - min_cnt={test_min_cnt} の場合: 実行エラー ({e})")

    # 元の分析処理
    analyze_and_print(all_score, all_lmcc, all_loc)


def main():
    for dataset_info in DATASETS:
        process_dataset(dataset_info)
    print("\n🎉 全ての相関分析が完了しました！")


if __name__ == "__main__":
    main()