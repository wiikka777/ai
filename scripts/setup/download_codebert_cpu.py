#!/usr/bin/env python3
"""
下载并缓存 CodeBERT 模型（CPU 环境）

默认缓存目录:
  C:\\Users\\Accio\\Desktop\\ai-test-master\\scripts\\setup\\hf_cache

用法:
  python scripts/setup/download_codebert_cpu.py
  python scripts/setup/download_codebert_cpu.py --cache_dir "D:\\hf_cache"
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Download and cache CodeBERT for CPU usage.")
    parser.add_argument(
        "--cache_dir",
        default=r"C:\Users\Accio\Desktop\ai-test-master\scripts\setup\hf_cache",
        help="Hugging Face cache directory",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    # Set cache env vars so downstream scripts can reuse the same model cache.
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir

    # Force CPU execution for this setup script.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    print(f"HF_HOME={cache_dir}")
    print(f"TRANSFORMERS_CACHE={cache_dir}")
    print("Device mode: CPU")

    try:
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        model_name = "microsoft/codebert-base-mlm"
        print(f"\n正在下载模型: {model_name}")
        print(f"目标缓存目录: {cache_dir}")

        print("\n[1/2] 下载模型权重...")
        AutoModelForMaskedLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        print("✓ 模型下载完成")

        print("[2/2] 下载分词器...")
        AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        print("✓ 分词器下载完成")

        print(f"\n缓存目录内容: {cache_dir}")
        if os.name != "nt":
            os.system(f'du -sh "{cache_dir}"')
        else:
            print("请在 PowerShell 中运行: Get-ChildItem -Recurse <cache_dir> | Measure-Object -Property Length -Sum")

    except Exception as e:
        print(f"❌ 错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
