import json
import sys
import traceback
from pathlib import Path

# ==========================================
# 1. パスの設定とデータセットの定義
# ==========================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = SCRIPT_DIR.parent.parent

BASE_DIR_MINE = PROJECTS_DIR / "detect-llm-hard-to-read" / "out"
BASE_DIR_PRIOR = PROJECTS_DIR / "lm-cc" / "results"

# 💡 変更点: 出力先のベースディレクトリを lm-cc プロジェクトの out/lm_cc に変更
OUTPUT_BASE_DIR = PROJECTS_DIR / "lm-cc" / "out" / "lm_cc"
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    {
        "name": "HumanEval",
        "prior_json": BASE_DIR_PRIOR / "humaneval-ier" / "entropy" / "entropies-CodeLlama-7b-hf-1.0.json",
        "my_dir": BASE_DIR_MINE / "humaneval",
        "parquet_format": "result_{task_id}.parquet"
    },
    {
        "name": "HumanEval-simplified",
        "prior_json": BASE_DIR_PRIOR / "humaneval-ier-simplified" / "entropy" / "entropies-CodeLlama-7b-hf-1.0.json",
        "my_dir": BASE_DIR_MINE / "humaneval_simplified-top60",
        "parquet_format": "result_{task_id}.parquet"
    },
    {
        "name": "XcodeEval(APR)",
        "prior_json": BASE_DIR_PRIOR / "xcodeeval" / "apr" / "entropy" / "entropies-CodeLlama-7b-hf-1.0.json",
        "my_dir": BASE_DIR_MINE / "xcodeeval_apr",
        "parquet_format": "result_{task_id}.parquet"
    },
    {
        "name": "XcodeEval-simplified(APR)",
        "prior_json": BASE_DIR_PRIOR / "xcodeeval" / "apr-simplified" / "entropy" / "entropies-CodeLlama-7b-hf-1.0.json",
        "my_dir": BASE_DIR_MINE / "xcodeeval_simplified-top50_apr",
        "parquet_format": "result_{task_id}.parquet"
    },
    {
        "name": "XcodeEval(Code Translation)",
        "prior_json": BASE_DIR_PRIOR / "xcodeeval" / "code_translation" / "entropy" / "entropies-CodeLlama-7b-hf-1.0.json",
        "my_dir": BASE_DIR_MINE / "xcodeeval_code_translation",
        "parquet_format": "result_{task_id}.parquet"
    },
    {
        "name": "XcodeEval-simplified(Code Translation)",
        "prior_json": BASE_DIR_PRIOR / "xcodeeval" / "code_translation-simplified" / "entropy" / "entropies-CodeLlama-7b-hf-1.0.json",
        "my_dir": BASE_DIR_MINE / "xcodeeval_simplified-top50_code_translation",
        "parquet_format": "result_{task_id}.parquet"
    }
]

# ==========================================
# 2. LM-CC モジュールのインポート
# ==========================================
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

try:
    from lm_cc.lm_cc import get_code_with_boundaries, CodeBlockProcessor, get_lmcc
except ImportError as e:
    print(f"❌ [致命的なエラー] lm_ccモジュールのインポートに失敗しました。")
    print(f"詳細: {e}")
    sys.exit(1)


# ==========================================
# 3. 処理ロジック
# ==========================================
def process_dataset(dataset_info, processor):
    ds_name = dataset_info.get("name", "Unknown")
    input_path = Path(dataset_info["prior_json"])

    print("\n" + "=" * 60)
    print(f"🚀 データセット処理開始: {ds_name}")
    print(f"📂 読み込み先: {input_path}")

    if not input_path.exists():
        print(f"  ⚠️ [スキップ] JSONファイルが存在しません。パスを確認してください。")
        return

    print(f"  🔍 JSONファイルをロードしています...")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"  ✅ ロード成功: 合計 {len(data)} 件のタスクが見つかりました。")
    except Exception as e:
        print(f"  ❌ [エラー] JSONの読み込みに失敗しました。\n詳細: {e}")
        return

    results = []
    success_count = 0
    error_count = 0

    for item in data:
        task_id = item.get("task_id", "Unknown")
        tokens = item.get("tokens")
        entropies = item.get("entropies")

        if not tokens or not entropies or len(tokens) != len(entropies):
            error_count += 1
            continue

        try:
            code_with_boundaries, _, start_end_tokens = get_code_with_boundaries(tokens, entropies, threshold=0.67)
            block_tree_dict = processor.parse_code_blocks(code_with_boundaries, tokens=tokens,
                                                          start_end_tokens=start_end_tokens)
            lm_cc_value = get_lmcc(block_tree_dict)

            results.append({
                "task_id": task_id,
                "lm_cc": lm_cc_value
            })
            success_count += 1

        except Exception as e:
            error_count += 1

    if not results:
        print(f"  ⚠️ [警告] 正常に処理されたデータが1件もありませんでした。保存をスキップします。")
        return

    # results ディレクトリからの相対パスを抽出して圧縮
    try:
        rel_path = input_path.relative_to(BASE_DIR_PRIOR)
        compressed_dir_name = str(rel_path.parent).replace("/", "_").replace("\\", "_")
    except ValueError:
        compressed_dir_name = input_path.parent.name

    out_dir = OUTPUT_BASE_DIR / compressed_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"lm_cc_{input_path.stem}.json"

    try:
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"  💾 保存完了: {out_file}")
        print(f"  📊 サマリー: 成功 {success_count} 件 / エラー {error_count} 件")
    except Exception as e:
        print(f"  ❌ [エラー] 結果のJSONファイルへの書き込みに失敗しました。\n詳細: {e}")


def main():
    print("⚙️ 初期化中 (CodeBlockProcessor)...")
    try:
        processor = CodeBlockProcessor()
    except Exception as e:
        print(f"❌ [致命的なエラー] CodeBlockProcessorの初期化に失敗しました。")
        print(f"詳細: {e}")
        sys.exit(1)

    for dataset_info in DATASETS:
        process_dataset(dataset_info, processor)

    print("\n🎉 全てのデータセットの処理が完了しました！")


if __name__ == "__main__":
    main()