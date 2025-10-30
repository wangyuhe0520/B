# 数据导入与分组准备
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import os  # 新增：用于文件夹操作

# 设置matplotlib不显示图形，只保存（关键设置，VS Code中生效）
plt.switch_backend('Agg')

# 解决 VS Code 中中文显示为方框的问题（新增代码）
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False  # 确保负号正常显示

# 新增：创建结果保存文件夹
output_dir = "analysis_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"已创建结果文件夹：{output_dir}")
else:
    print(f"结果文件夹已存在：{output_dir}")

# 导入清洗后的数据（含“索赔金额”）
df = pd.read_excel("F:/1/prediction_results_fixed_updated (2).xlsx")

# 根据十分位分组法和EDA分析定义全部分组区间（前9组+10组拆分的4个子组，共13组）
groups = [
    (0.84, 36.49),          # 1组
    (36.56, 70.21),        # 2组
    (70.29, 104.31),        # 3组
    (104.36, 147.47),        # 4组
    (147.48, 198.06),        # 5组
    (198.24, 273.57),        # 6组
    (273.75, 350.16),        # 7组
    (351.02, 482),       # 8组
    (482.32, 675.24),      # 9组
    (675.25, 800),      # 10-1组（高赔付子组1）
    (800, 1000),      # 10-2组（高赔付子组2）
    (1000, 1500),      # 10-3组（高赔付子组3）
    (1500, 3000)       # 10-4组（高赔付子组4）
]


# 制定每组标注阈值（核心：按比例分位数计算）
# 存储每组阈值的字典：key为组名，value为(Q85, Q97)
thresholds = {}

for i, (low, high) in enumerate(groups, 1):
    # 筛选当前组的运单
    group_data = df[(df["payment_real_money"] >= low) & (df["payment_real_money"] <= high)]
    if group_data.empty:
        print(f"警告：第{i}组无数据，需检查分组区间")
        continue
    
    # 提取plan_delv_to_real_delv_diff并排序（从大到小）
    diffs = group_data["plan_delv_to_real_delv_diff"].sort_values(ascending=False).values
    n = len(diffs)
    
    # 计算85%分位（合理诉求下限）和97%分位（严重超额上限）
    # 注：用nearest方法确保分位值为实际存在的差额
    q85 = np.percentile(diffs, 15, method="nearest")  # 前85%的最小差额（第15百分位）
    q97 = np.percentile(diffs, 3, method="nearest")   # 前97%的最小差额（第3百分位）
    
    # 阈值合理性检查
    group_name = f"10-{i-9}" if i > 9 else f"{i}"  # 提前定义group_name用于警告信息
    if q85 < q97:
        print(f"警告：第{group_name}组阈值异常（q85={q85} < q97={q97}）")
    
    # 存储阈值（组名区分原组和子组）
    thresholds[group_name] = (q85, q97)
    
    # 打印每组阈值，验证合理性
    print(f"第{group_name}组（{low}~{high}元）：合理≥{q85}，偏高[{q97}, {q85})，超额<{q97}")


# 生成分组名称列表（用于验证函数）
group_names = [f"10-{i-9}" if i > 9 else f"{i}" for i in range(1, len(groups)+1)]


# 用规则标注全量数据
# 定义标注函数
def label_risk(row):
    amount = row["payment_real_money"]
    diff = row["plan_delv_to_real_delv_diff"]
    
    # 匹配所属分组
    for i, (low, high) in enumerate(groups, 1):
        if low <= amount <= high:
            group_name = f"10-{i-9}" if i > 9 else f"{i}"
            # 处理空组情况
            if group_name not in thresholds:
                return "分组无数据（异常）"
            q85, q97 = thresholds[group_name]
            if diff >= q85:
                return "合理诉求"
            elif diff >= q97:
                return "诉求偏高"
            else:
                return "严重超额"
    return "未分组（异常）"  # 应对极端值

# 应用标注函数
df["风险标注结果"] = df.apply(label_risk, axis=1)



# ----------------------
# 2. 验证条件2：阈值趋势（单调递减）
# ----------------------
def verify_condition2():
    print("----- 验证条件2：高赔付组阈值更宽松 -----")
    # 提取每组的区间上限和合理诉求阈值（q85）
    trend_data = []
    for i, (low, high) in enumerate(groups):
        group_name = group_names[i]
        # 跳过无数据的分组
        if group_name not in thresholds:
            continue
        q85, _ = thresholds[group_name]
        trend_data.append({
            "分组": group_name,
            "区间上限": high,
            "合理诉求阈值(q85)": q85
        })
    trend_df = pd.DataFrame(trend_data)
    
    # 按区间上限升序排列（确保顺序正确）
    trend_df = trend_df.sort_values("区间上限").reset_index(drop=True)
    print("各组阈值趋势数据：")
    print(trend_df[["分组", "区间上限", "合理诉求阈值(q85)"]])
    
    # 绘制折线图并保存到文件夹
    plt.figure(figsize=(12, 6))
    plt.plot(trend_df["区间上限"], trend_df["合理诉求阈值(q85)"], 
             marker="o", color="darkred", linewidth=2, markersize=8)
    plt.xticks(trend_df["区间上限"], trend_df["分组"], rotation=45)  # x轴显示分组名
    plt.xlabel("payment_real_money区间上限（元）")
    plt.ylabel("合理诉求阈值（plan_delv_to_real_delv_diff，元）")
    plt.title("条件2验证：payment_real_money与合理诉求阈值的趋势（应单调递减）")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "条件2验证_阈值趋势图.png"))  # 保存到指定文件夹
    plt.close()  # 关闭图像释放资源
    
    # 验证是否单调递减
    if len(trend_df) <= 1:
        print("⚠️ 有效分组不足，无法验证阈值趋势")
    else:
        is_decreasing = all(trend_df["合理诉求阈值(q85)"].iloc[i] >= trend_df["合理诉求阈值(q85)"].iloc[i+1] 
                            for i in range(len(trend_df)-1))
        if is_decreasing:
            print("✅ 条件2验证通过：阈值随赔付金额升高单调递减")
        else:
            print("❌ 条件2验证不通过：存在阈值未随金额升高而降低的分组")


# ----------------------
# 3. 验证条件3：同组内差额集中
# ----------------------
def verify_condition3(std_threshold=300):
    print("\n----- 验证条件3：同组内同一类型差额集中 -----")
    # 按分组验证“合理诉求”的标准差和核密度图
    for i, (low, high) in enumerate(groups):
        group_name = group_names[i]
        # 跳过无数据的分组
        if group_name not in thresholds:
            print(f"⚠️ {group_name}无数据，跳过验证")
            continue
        # 筛选该组的合理诉求运单
        group_data = df[(df["payment_real_money"] >= low) & (df["payment_real_money"] <= high)]
        reasonable_data = group_data[group_data["风险标注结果"] == "合理诉求"]
        if len(reasonable_data) < 10:  # 样本量过少则跳过
            print(f"⚠️ {group_name}合理诉求样本量不足，跳过验证")
            continue
        
        # 计算标准差
        diff_std = reasonable_data["plan_delv_to_real_delv_diff"].std()
        # 绘制核密度图并保存
        plt.figure(figsize=(8, 4))
        sns.kdeplot(reasonable_data["plan_delv_to_real_delv_diff"], fill=True, color="teal")
        plt.axvline(x=thresholds[group_name][0], color="red", linestyle="--", 
                   label=f"合理诉求阈值(q85): {thresholds[group_name][0]}")
        plt.title(f"{group_name}（{low}~{high}元）合理诉求差额分布")
        plt.xlabel("plan_delv_to_real_delv_diff（元）")
        plt.ylabel("密度")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{group_name}_合理诉求密度图.png"))  # 保存到指定文件夹
        plt.close()  # 关闭图像释放资源
        
        # 验证标准
        if diff_std < std_threshold:
            print(f"✅ {group_name}：标准差={diff_std:.2f} < {std_threshold}，差额集中")
        else:
            print(f"❌ {group_name}：标准差={diff_std:.2f} ≥ {std_threshold}，差额分散，需优化")


# ----------------------
# 4. 验证条件4：合理密集vs超额稀疏
# ----------------------
def verify_condition4():
    print("\n----- 验证条件4：合理诉求密集，严重超额稀疏 -----")
    for i, (low, high) in enumerate(groups):
        group_name = group_names[i]
        # 跳过无数据的分组
        if group_name not in thresholds:
            print(f"⚠️ {group_name}无数据，跳过验证")
            continue
        # 筛选该组的合理诉求和严重超额运单
        group_data = df[(df["payment_real_money"] >= low) & (df["payment_real_money"] <= high)]
        reasonable_data = group_data[group_data["风险标注结果"] == "合理诉求"]
        excess_data = group_data[group_data["风险标注结果"] == "严重超额"]
        
        if len(reasonable_data) < 10 or len(excess_data) < 3:  # 样本量不足则跳过
            print(f"⚠️ {group_name}样本量不足（合理诉求={len(reasonable_data)}，超额={len(excess_data)}），跳过验证")
            continue
        
        # 1. 散点图验证并保存
        plt.figure(figsize=(10, 5))
        plt.scatter(reasonable_data["payment_real_money"], reasonable_data["plan_delv_to_real_delv_diff"], 
                   label="合理诉求", s=20, alpha=0.6, color="blue")
        plt.scatter(excess_data["payment_real_money"], excess_data["plan_delv_to_real_delv_diff"], 
                   label="严重超额", s=20, alpha=0.8, color="red", marker="x")
        plt.axhline(y=thresholds[group_name][1], color="purple", linestyle="--", 
                   label=f"严重超额阈值(q97): {thresholds[group_name][1]}")
        plt.title(f"{group_name}（{low}~{high}元）差额分布散点图")
        plt.xlabel("payment_real_money（元）")
        plt.ylabel("plan_delv_to_real_delv_diff（元）")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{group_name}_散点图.png"))  # 保存到指定文件夹
        plt.close()  # 关闭图像释放资源
        
        # 2. 直方图对比并保存
        plt.figure(figsize=(12, 5))
        # 合理诉求直方图
        plt.subplot(1, 2, 1)
        plt.hist(reasonable_data["plan_delv_to_real_delv_diff"], bins=15, color="blue", alpha=0.7)
        plt.axvline(x=thresholds[group_name][0], color="red", linestyle="--")
        plt.title(f"{group_name}合理诉求差额分布")
        plt.xlabel("plan_delv_to_real_delv_diff（元）")
        plt.ylabel("频数")
        
        # 严重超额直方图
        plt.subplot(1, 2, 2)
        plt.hist(excess_data["plan_delv_to_real_delv_diff"], bins=10, color="red", alpha=0.7)
        plt.axvline(x=thresholds[group_name][1], color="purple", linestyle="--")
        plt.title(f"{group_name}严重超额差额分布")
        plt.xlabel("plan_delv_to_real_delv_diff（元）")
        plt.ylabel("频数")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{group_name}_直方图对比.png"))  # 保存到指定文件夹
        plt.close()  # 关闭图像释放资源
        
        # 验证标准（定性描述）
        print(f"✅ {group_name}：请结合图表确认是否满足“合理密集、超额稀疏”（合理点聚集，超额点分散）")


# 执行验证函数
verify_condition2()
verify_condition3()
verify_condition4()


# 结果整理与输出（保存到指定文件夹）
# 保存标注结果（含所有字段）
df.to_csv(os.path.join(output_dir, "附件1_风险标注结果.csv"), index=False)
print(f"标注结果已保存至：{os.path.join(output_dir, '附件1_风险标注结果.csv')}")

# 保存分组阈值表（用于论文说明）
thresholds_df = pd.DataFrame([(k, v[0], v[1]) for k, v in thresholds.items()],
                             columns=["分组", "合理诉求阈值（≥）", "严重超额阈值（<）"])
thresholds_df.to_excel(os.path.join(output_dir, "风险标注阈值表.xlsx"), index=False)
print(f"分组阈值表已保存至：{os.path.join(output_dir, '风险标注阈值表.xlsx')}")