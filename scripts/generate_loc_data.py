import json
import sys
from pathlib import Path

# ==========================================
# 1. パスの設定
# ==========================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = SCRIPT_DIR.parent.parent

# データの入力元ディレクトリ
DATA_BASE_DIR = PROJECTS_DIR / "detect-llm-hard-to-read" / "data"

# 💡 変更点: 出力先ディレクトリを out/loc に変更
OUTPUT_BASE_DIR = PROJECTS_DIR / "lm-cc" / "out" / "loc"

DATASETS = [
    {"data_dir": "humaneval", "out_dir": "humaneval-ier"},
    {"data_dir": "humaneval_simplified-top60", "out_dir": "humaneval-ier-simplified-top60"},
    {"data_dir": "xcodeeval", "out_dir": "xcodeeval_apr"},
    {"data_dir": "xcodeeval_simplified-top50", "out_dir": "xcodeeval_apr-simplified-top50"},
    {"data_dir": "xcodeeval", "out_dir": "xcodeeval_code_translation"},
    {"data_dir": "xcodeeval_simplified-top50", "out_dir": "xcodeeval_translation-simplified-top50"}
]

# ==========================================
# 2. 先行研究の metrics.py のインポート
# ==========================================
utils_dir = SCRIPT_DIR / "utils"
if str(utils_dir) not in sys.path:
    sys.path.append(str(utils_dir))

try:
    from scripts.utils.metrics import get_code_loc
except ImportError as e:
    print(f"❌ [エラー] utils/metrics.py から get_code_loc をインポートできませんでした。")
    print(f"詳細: {e}")
    sys.exit(1)


# ==========================================
# 3. 処理ロジック
# ==========================================
def process_dataset(dataset_info):
    data_dir_name = dataset_info["data_dir"]
    out_dir_name = dataset_info["out_dir"]

    input_dir = DATA_BASE_DIR / data_dir_name
    output_dir = OUTPUT_BASE_DIR / out_dir_name
    output_file = output_dir / "loc.json"

    print(f"\n🚀 処理開始: {data_dir_name}")

    if not input_dir.exists():
        print(f"  ⚠️ [スキップ] ディレクトリが存在しません: {input_dir}")
        return

    loc_results = {}
    success_count = 0
    error_count = 0

    python_files = list(input_dir.rglob("*.py"))
    print(f"  🔍 {len(python_files)} 件のPythonファイルが見つかりました。")

    for py_file in python_files:
        task_id = py_file.stem
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                code_content = f.read()
            loc_value = get_code_loc(code_content)
            loc_results[task_id] = loc_value
            success_count += 1
        except Exception as e:
            error_count += 1

    if not loc_results:
        print("  ⚠️ 保存するLOCデータがありません。")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(loc_results, f, indent=4)

    print(f"  💾 保存完了: {output_file}")
    print(f"  📊 サマリー: 成功 {success_count} 件 / エラー {error_count} 件")


def main():
    print("⚙️ LOC算出スクリプトを開始します...")
    for dataset_info in DATASETS:
        process_dataset(dataset_info)
    print("\n🎉 全ての処理が完了しました！")


if __name__ == "__main__":
    main()