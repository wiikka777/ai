#!/usr/bin/env python3
"""
验证所有数据路径和配置是否正确
"""

import os
import json
import sys

def check_paths():
    """检查所有关键文件的路径"""
    print("="*80)
    print("数据路径验证")
    print("="*80)
    
    # 确定项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = script_dir
    
    print(f"\n项目根目录: {project_root}\n")
    
    # 检查数据文件
    data_files = {
        'AI samples': os.path.join(project_root, 'data', 'raw', 'ai.json'),
        'Human samples (before 2022)': os.path.join(project_root, 'data', 'raw', 'slice_before_2022_11_01_5000.json'),
        'Full dataset (last_success)': os.path.join(project_root, 'data', 'raw', 'smartbeans_submission_last_success.json'),
        'After 2024 samples': os.path.join(project_root, 'data', 'raw', 'slice_after_2024_06_01_latest_5000.json'),
    }
    
    print("数据文件检查:")
    print("-"*80)
    
    all_exist = True
    for name, path in data_files.items():
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        
        if exists:
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"{status} {name:40} | {size_mb:8.2f} MB")
        else:
            print(f"{status} {name:40} | 文件不存在")
            all_exist = False
    
    # 检查模型依赖
    print("\n模型依赖检查:")
    print("-"*80)
    
    model_paths = {
        'GPTZero-main': [
            '/user/zhuohang.yu/u24922/exam/GPTZero-main/GPTZero-main/model.py',
            os.path.expanduser('~/GPTZero-main/GPTZero-main/model.py'),
            './GPTZero-main/GPTZero-main/model.py',
        ],
        'DetectGPT-main': [
            '/user/zhuohang.yu/u24922/exam/DetectGPT-main/model.py',
            os.path.expanduser('~/DetectGPT-main/model.py'),
            './DetectGPT-main/model.py',
        ]
    }
    
    for model_name, paths in model_paths.items():
        found = False
        for path in paths:
            if os.path.exists(path):
                print(f"✅ {model_name:20} | 找到: {path}")
                found = True
                break
        if not found:
            print(f"⚠️  {model_name:20} | 未找到（如果要用该方法需要配置）")
            print(f"    尝试的路径: {', '.join(paths)}")
    
    # 检查HuggingFace缓存
    print("\nHuggingFace缓存配置:")
    print("-"*80)
    
    hf_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
    print(f"HF_HOME: {hf_home}")
    print(f"存在: {'✅' if os.path.exists(hf_home) else '❌'}")
    
    # 总结
    print("\n" + "="*80)
    print("总结")
    print("="*80)
    
    if all_exist:
        print("\n✅ 所有必要的数据文件都存在！可以运行实验。")
        print("\n快速开始:")
        print("  python scripts/experiments/comparison_experiment_v2.py --part1_only")
        print("\n其他选项:")
        print("  --part1_only          仅运行Part 1（有标签评估）")
        print("  --part2_only          仅运行Part 2（44k分析）")
        print("  --sample_size 5000    在Part 2中采样5000个样本（加速）")
        print("  --device cuda         使用GPU")
        return 0
    else:
        print("\n❌ 缺少必要的数据文件！请检查以下文件:")
        for name, path in data_files.items():
            if not os.path.exists(path):
                print(f"  - {name}: {path}")
        return 1


if __name__ == "__main__":
    sys.exit(check_paths())
