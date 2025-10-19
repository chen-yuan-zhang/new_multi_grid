import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":

    datasets = ["result.csv"]

    for dataset in datasets:
        # 读取数据
        df = pd.read_csv(dataset)

        # 需要对比的“帮助策略”列名与展示名
        help_cols = [
            ("steps_with_help_uniform", "Uniform"),
            ("steps_with_help_greedy",  "Greedy"),
            ("steps_with_help_gr",  "Goal Recognition"),
            ("steps_with_help_upperbound", "Upperbound"),
            ("steps_with_help_paper", "paper"),
            ("steps_with_help_paper2", "paper2")

            
        ]

        # 基线列名
        base_col = "steps_baseline"

        # 若缺少 baseline，直接跳过
        if base_col not in df.columns:
            print(f"[WARN] {dataset}: missing '{base_col}', skip.")
            continue

        # 也顺手打印 baseline 的均值
        avg_steps = float(np.nanmean(df[base_col].astype(float)))
        print(f"Dataset: {dataset}")
        print(f"Baseline avg steps: {avg_steps:.2f}")
        
        # 逐个帮助策略与 baseline 对比
        for col, label in help_cols:
            print("-" * 60)
            if col not in df.columns:
                print(f"[INFO] {dataset}: '{col}' not found; skip this help variant.")
                continue

            # 向量化清洗
            s  = pd.to_numeric(df[base_col], errors="coerce")
            sh = pd.to_numeric(df[col],       errors="coerce")

            # 过滤缺失
            mask = s.notna() & sh.notna()
            s, sh = s[mask].astype(float), sh[mask].astype(float)

            # 计算差值/比例/成功率
            diff = s - sh
            succ_rate = float((diff > 0).mean()) if len(diff) else 0.0

            # 仅在 s>0 时计算相对比例
            mask_pos = s > 0
            mean_ratio = float(((diff[mask_pos]) / s[mask_pos]).mean()) if mask_pos.any() else 0.0

            mean_diff = float(diff.mean()) if len(diff) else 0.0
            avg_steps_help = float(sh.mean()) if len(sh) else float("nan")

            # 输出结果（英文）
            print(
                f"[{label}]  Δsteps(mean)={mean_diff:.2f},  "
                f"Mean ratio={mean_ratio:.2%},  "
                f"Success(reduced)={succ_rate:.2%},  "
                f"Avg steps (help)={avg_steps_help:.2f}"
            )

