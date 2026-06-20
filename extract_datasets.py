import json
import shutil
from pathlib import Path

# --- 設定 ---
# 先行研究のデータセットが入っているルートディレクトリ
DATASET_ROOT = Path("dataset")
# 抽出した物理ファイルを保存する出力先ディレクトリ
OUTPUT_ROOT = Path("data")


def extract_humaneval():
    print("--- Extracting HumanEval ---")
    for category in ["humaneval", "humaneval_simplified", "humaneval_simplified-top60"]:
        src_dir = DATASET_ROOT / category
        out_dir = OUTPUT_ROOT / category
        out_dir.mkdir(parents=True, exist_ok=True)

        if not src_dir.exists():
            print(f"[Skip] {src_dir} not found.")
            continue

        seen_problems = set()

        for problem_dir in src_dir.glob("HumanEval_*"):
            main_py = problem_dir / "main.py"
            if main_py.exists():
                dir_name = problem_dir.name
                base_problem_id = dir_name.split("__")[0]

                if base_problem_id in seen_problems:
                    continue

                seen_problems.add(base_problem_id)

                out_file = out_dir / f"{base_problem_id}.py"
                shutil.copy(main_py, out_file)

        print(f"✅ {category} の抽出が完了しました。（ユニーク問題数: {len(seen_problems)}）")


def extract_xcodeeval_apr():
    print("--- Extracting xCodeEval APR ---")
    apr_dir = DATASET_ROOT / "xcodeeval" / "apr"

    # 出力先ディレクトリを分ける
    out_dir_original = OUTPUT_ROOT / "xcodeeval" / "apr"
    out_dir_simplified = OUTPUT_ROOT / "xcodeeval_simplified" / "apr"
    out_dir_simplified_top50 = OUTPUT_ROOT / "xcodeeval_simplified-top50" / "apr"
    out_dir_original.mkdir(parents=True, exist_ok=True)
    out_dir_simplified.mkdir(parents=True, exist_ok=True)
    out_dir_simplified_top50.mkdir(parents=True, exist_ok=True)

    # 読み込むファイルと、その出力先の対応付け
    file_mapping = [
        ("xcodeeval_apr_test_filtered.jsonl", out_dir_original),
        ("xcodeeval_apr_test_filtered-simplified.jsonl", out_dir_simplified),
        ("xcodeeval_apr_test_filtered-simplified-top50.jsonl", out_dir_simplified_top50)
    ]

    for filename, out_dir in file_mapping:
        filepath = apr_dir / filename
        if not filepath.exists():
            print(f"[Skip] {filepath} not found.")
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                uid = data.get("bug_code_uid", "unknown_uid")
                code = data.get("bug_source_code", "")

                if code:
                    # フォルダが分かれたので、ファイル名はuidのみでOK
                    out_file = out_dir / f"{uid}.py"
                    out_file.write_text(code, encoding="utf-8")

        print(f"✅ {filename} を {out_dir.name}/ に抽出しました。")


def extract_xcodeeval_ct():
    print("--- Extracting xCodeEval Code Translation ---")
    ct_dir = DATASET_ROOT / "xcodeeval" / "code_translation"

    # 出力先ディレクトリを分ける
    out_dir_original = OUTPUT_ROOT / "xcodeeval" / "code_translation"
    out_dir_simplified = OUTPUT_ROOT / "xcodeeval_simplified" / "code_translation"
    out_dir_simplified_top50 = OUTPUT_ROOT / "xcodeeval_simplified-top50" / "code_translation"
    out_dir_original.mkdir(parents=True, exist_ok=True)
    out_dir_simplified.mkdir(parents=True, exist_ok=True)
    out_dir_simplified_top50.mkdir(parents=True, exist_ok=True)

    # 読み込むファイルと、その出力先の対応付け
    file_mapping = [
        ("xcodeeval_codetranslation_test_filtered.jsonl", out_dir_original),
        ("xcodeeval_codetranslation_test_filtered_simplified.jsonl", out_dir_simplified),
        ("xcodeeval_codetranslation_test_filtered_simplified-top50.jsonl", out_dir_simplified_top50)
    ]

    for filename, out_dir in file_mapping:
        filepath = ct_dir / filename
        if not filepath.exists():
            print(f"[Skip] {filepath} not found.")
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                uid = data.get("code_uid", "unknown_uid")
                code = data.get("source_code", data.get("code", ""))

                lang = data.get("source_lang", data.get("lang", "")).lower()
                ext = ".py" if "python" in lang else ".java" if "java" in lang else ".txt"

                if code:
                    out_file = out_dir / f"{uid}{ext}"
                    out_file.write_text(code, encoding="utf-8")

        print(f"✅ {filename} を {out_dir.name}/ に抽出しました。")


if __name__ == "__main__":
    print("🚀 データセット抽出を開始します...")
    extract_humaneval()
    extract_xcodeeval_apr()
    extract_xcodeeval_ct()
    print("🎉 すべての抽出が完了しました！")