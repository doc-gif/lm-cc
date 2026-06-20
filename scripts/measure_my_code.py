import torch
from lm_cc import get_code_lmcc, CodeBlockProcessor, TokenEntropyCalculator

def main():
    # 💡 Mac (Apple Silicon) 向けの最適化設定
    # MPS (Metal Performance Shaders) が使える場合はそれを使用し、無理ならCPUにフォールバック
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"✅ 使用デバイス: {device}")

    # 計測したいソースコード
    sample_code = """
from typing import *
def check_dict_case(dict):
    if len(dict.keys()) == 0:
        return False
    else:
        state = "start"
        for key in dict.keys():

            if isinstance(key, str) == False:
                state = "mixed"
                break
            if state == "start":
                if key.isupper():
                    state = "upper"
                elif key.islower():
                    state = "lower"
                else:
                    break
            elif (state == "upper" and not key.isupper()) or (state == "lower" and not key.islower()):
                    state = "mixed"
                    break
            else:
                break
        return state == "upper" or state == "lower" 

"""

    print("⏳ モデルを読み込んでいます... (初回はダウンロードに時間がかかります)")
    processor = CodeBlockProcessor()
    
    # 7BモデルをMacのメモリに収めるため、必ず float16 で読み込む
    calculator = TokenEntropyCalculator(
        model_name="codellama/CodeLlama-7b-hf",
        cache_dir="./hf_cache",  # ダウンロード先を明示
        device_map=device,       
        float_type=torch.float16 
    )

    print("📊 LM-CC を計測中...")
    cc_score = get_code_lmcc(sample_code, calculator=calculator, processor=processor)
    
    print("\n" + "="*30)
    print(f"🎯 算出された LM-CC: {cc_score:.4f}")
    print("="*30)

if __name__ == "__main__":
    main()
