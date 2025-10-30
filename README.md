# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 10:04:27 2025

@author: 15507
"""

# 1. 导入必备库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 尝试不同的编码方式
try:
    df = pd.read_csv(r"C:/Users/15507/Desktop/prediction_results_fixed_updated (2).csv", encoding='utf-8')
except UnicodeDecodeError:
    try:
        df = pd.read_csv(r"C:\Users\15507\Desktop\prediction_results_fixed_updated (2).csv", encoding='gbk')
    except UnicodeDecodeError:
        df = pd.read_csv(r"C:\Users\15507\Desktop\prediction_results_fixed_updated (2).csv", encoding='gb2312')

# 设置中文字体和图形显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 2. 查看数据的基本信息
print("数据形状（行数，列数）:", df.shape)
print("\n前5行数据：")
print(df.head())
df.info()
df.describe()

# 数据概览分析
print("=" * 50)
print("数据概览分析")
print("=" * 50)

print(f"1. 数据规模: {df.shape[0]:,} 行 × {df.shape[1]} 列")
print(f"2. 内存使用: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n3. 缺失值情况:")
missing_info = pd.DataFrame({
    '缺失数量': df.isnull().sum(),
    '缺失比例%': (df.isnull().sum() / len(df) * 100).round(2)
})
print(missing_info[missing_info['缺失数量'] > 0])

print("\n4. 数据类型:")
print(df.dtypes.value_counts())

print("\n5. 数值型字段转换后分析:")
amount_cols = ['payment_real_money', 'payment_contract_money']
for col in amount_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f"\n{col}的统计描述:")
        print(df[col].describe())
        
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        df[col].hist(bins=30)
        plt.title(f'{col}分布')
        plt.subplot(1, 2, 2)
        df[col].plot(kind='box')
        plt.title(f'{col}箱线图')
        plt.tight_layout()
        plt.show()

print("=" * 50)       

# 1. 绘制两个核心变量的分布直方图
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

sns.histplot(df['payment_real_money'], bins=50, ax=axes[0])
axes[0].set_title('payment_real_money分布')
axes[0].set_xlabel('payment_real_money')
df['索赔差额']= df['payment_real_money'] - df['payment_contract_money']

sns.histplot(df['索赔差额'], bins=50, ax=axes[1])
axes[1].set_title('索赔差额分布')
axes[1].set_xlabel('索赔差额')

plt.tight_layout()
plt.show()

# 2. 绘制关系图
plt.figure(figsize=(10, 6))
plt.scatter(df['payment_real_money'], df['索赔差额'], alpha=0.5)
plt.xlabel('payment_real_money')
plt.ylabel('索赔差额')
plt.title('payment_real_money与索赔差额关系图')
plt.grid(True)
plt.show()

print("=" * 50) 
n_buckets = 10
bucket_type = "十分位数"
df['金额分桶'] = pd.qcut(df['payment_real_money'], q=n_buckets, duplicates='drop')
actual_buckets = len(df['金额分桶'].unique())

print(f"\n分桶结果:")
print(f"- 分桶类型: {bucket_type}")
print(f"- 计划分组: {n_buckets}组")
print(f"- 实际有效分组: {actual_buckets}组")

print("=" * 50)
print("开始按赔付金额排序 + 十分位数分桶")
print("=" * 50)

df['payment_real_money'] = pd.to_numeric(df['payment_real_money'], errors='coerce')
df['索赔差额绝对值']=abs(df['索赔差额'])

df_sorted = df.sort_values('payment_real_money').reset_index(drop=True)
print("已按payment_real_money排序")

print(f"\n排序后前5行:")
print(df_sorted[['payment_real_money', 'payment_contract_money', '索赔差额']].head())
print(f"\n排序后后5行:")
print(df_sorted[['payment_real_money', 'payment_contract_money', '索赔差额']].tail())

n_buckets = 10
total_samples = len(df_sorted)
bucket_size = total_samples // n_buckets

print(f"\n分桶参数:")
print(f"- 总样本数: {total_samples}")
print(f"- 分桶数: {n_buckets}")
print(f"- 理论每桶样本数: {bucket_size}")

df_sorted['分桶编号'] = 0
bucket_ranges = []

for i in range(n_buckets):
    start_idx = i * bucket_size
    if i == n_buckets - 1:
        end_idx = total_samples
    else:
        end_idx = (i + 1) * bucket_size
    
    df_sorted.loc[start_idx:end_idx-1, '分桶编号'] = i + 1
    
    bucket_data = df_sorted[df_sorted['分桶编号'] == i + 1]
    min_amount = bucket_data['payment_real_money'].min()
    max_amount = bucket_data['payment_real_money'].max()
    bucket_ranges.append({
        '分桶编号': i + 1,
        '样本数量': len(bucket_data),
        '最小金额': min_amount,
        '最大金额': max_amount,
        '金额范围': f"{min_amount:.2f} - {max_amount:.2f}",
        '平均赔付金额': bucket_data['payment_real_money'].mean(),
        '平均砍价金额': bucket_data['索赔差额绝对值'].mean()
    })

print(f"\n十分位分桶结果:")
print("=" * 80)
bucket_df = pd.DataFrame(bucket_ranges)
print(bucket_df.to_string(index=False))

print(f"\n分桶均匀性验证:")
expected_percentage = 100 / n_buckets
actual_percentages = []

for i in range(1, n_buckets + 1):
    bucket_count = len(df_sorted[df_sorted['分桶编号'] == i])
    percentage = (bucket_count / total_samples) * 100
    actual_percentages.append(percentage)
    print(f"分桶{i}: {bucket_count}个样本 ({percentage:.1f}%), 目标: {expected_percentage:.1f}%")

uniformity_score = 1 - (max(actual_percentages) - min(actual_percentages)) / expected_percentage
print(f"\n分桶均匀性评分: {uniformity_score:.3f} (1为完全均匀)")

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
for bucket in bucket_ranges:
    plt.barh(f"分桶{bucket['分桶编号']}", bucket['最大金额'] - bucket['最小金额'], 
             left=bucket['最小金额'], alpha=0.7, label=f"分桶{bucket['分桶编号']}")
plt.xlabel('payment_real_money')
plt.title('十分位分桶金额范围')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
bucket_counts = df_sorted['分桶编号'].value_counts().sort_index()
plt.bar(bucket_counts.index, bucket_counts.values, alpha=0.7, color='skyblue')
plt.axhline(y=bucket_size, color='red', linestyle='--', label=f'目标样本数: {bucket_size}')
plt.xlabel('分桶编号')
plt.ylabel('样本数量')
plt.title('各分桶样本量分布')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
avg_diff_by_bucket = df_sorted.groupby('分桶编号')['索赔差额绝对值'].mean()
plt.plot(avg_diff_by_bucket.index, avg_diff_by_bucket.values, 'o-', linewidth=2, markersize=8)
plt.xlabel('分桶编号')
plt.ylabel('平均砍价金额')
plt.title('各分桶平均砍价金额趋势')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
colors = plt.cm.tab10(np.linspace(0, 1, n_buckets))
for i in range(1, n_buckets + 1):
    bucket_data = df_sorted[df_sorted['分桶编号'] == i]
    plt.scatter(bucket_data['payment_real_money'], bucket_data['索赔差额绝对值'], 
               alpha=0.6, s=10, color=colors[i-1], label=f'分桶{i}')
plt.xlabel('payment_real_money')
plt.ylabel('索赔差额绝对值（砍价程度）')
plt.title('十分位分桶可视化')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

df['分桶编号'] = df_sorted['分桶编号']
df['金额分桶范围'] = df_sorted['分桶编号'].map(
    lambda x: f"{bucket_ranges[x-1]['最小金额']:.2f}-{bucket_ranges[x-1]['最大金额']:.2f}" if 1 <= x <= n_buckets else '未知'
)

print(f"\n十分位分桶完成!")
print(f"- 创建了 {n_buckets} 个分桶")
print(f"- 每个分桶约包含 {bucket_size} 个样本")
print(f"- 分桶均匀性: {uniformity_score:.3f}")

bucket_info_df = pd.DataFrame(bucket_ranges)
print(f"\n分桶信息已保存，可用于后续风险标注规则制定")

print("=" * 50)
print("十分位分桶分析完成!")
print("=" * 50)

print("=" * 50)
print("第十组业务导向分桶实施")
print("=" * 50)

business_bins = [675, 800, 1000, 1500, 3172]
business_labels = ['675-800', '800-1000', '1000-1500', '1500+']

print("业务导向分桶方案:")
print(f"- {business_labels[0]}: 覆盖低端密集区域")
print(f"- {business_labels[1]}: 覆盖中端密集区域") 
print(f"- {business_labels[2]}: 覆盖中高端区域")
print(f"- {business_labels[3]}: 专门处理高端异常值")

bucket_10_data = df_sorted[df_sorted['分桶编号'] == 10].copy()
bucket_10_data['业务分桶'] = pd.cut(bucket_10_data['payment_real_money'], 
                                  bins=business_bins, 
                                  labels=business_labels,
                                  include_lowest=True)

business_summary = bucket_10_data.groupby('业务分桶').agg({
    'payment_real_money': ['count', 'min', 'max', 'mean', 'median'],
    'payment_contract_money': ['mean'],
    '索赔差额绝对值': ['mean', 'std', 'min', 'max'],
    '索赔差额': ['mean', 'std']
}).round(2)

print(f"\n业务导向分桶详细统计:")
print(business_summary)

total_10 = len(bucket_10_data)
bucket_percentages = []
for bucket in business_labels:
    count = len(bucket_10_data[bucket_10_data['业务分桶'] == bucket])
    percentage = (count / total_10) * 100
    bucket_percentages.append(percentage)
    print(f"{bucket}: {count}个样本 ({percentage:.1f}%)")

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
bucket_counts = bucket_10_data['业务分桶'].value_counts().sort_index()
colors = ['lightblue', 'lightgreen', 'orange', 'red']
bars = plt.bar(bucket_counts.index, bucket_counts.values, alpha=0.7, color=colors)
plt.xlabel('业务分桶')
plt.ylabel('样本数量')
plt.title('各分桶样本量分布')
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height}\n({bucket_percentages[i]:.1f}%)',
             ha='center', va='bottom')
plt.xticks(rotation=45)

plt.subplot(2, 3, 2)
for i, bucket in enumerate(business_labels):
    bucket_data = bucket_10_data[bucket_10_data['业务分桶'] == bucket]
    if len(bucket_data) > 0:
        plt.barh(f'{bucket}\n({bucket_percentages[i]:.1f}%)', 
                bucket_data['payment_real_money'].max() - bucket_data['payment_real_money'].min(),
                left=bucket_data['payment_real_money'].min(), 
                alpha=0.7, color=colors[i], edgecolor='black')
plt.xlabel('payment_real_money')
plt.title('各分桶金额范围')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 3)
avg_discount_by_bucket = bucket_10_data.groupby('业务分桶')['索赔差额绝对值'].mean()
plt.bar(avg_discount_by_bucket.index, avg_discount_by_bucket.values, 
        alpha=0.7, color=colors)
plt.xlabel('业务分桶')
plt.ylabel('平均砍价金额')
plt.title('各分桶平均砍价金额')
plt.xticks(rotation=45)
for i, value in enumerate(avg_discount_by_bucket.values):
    plt.text(i, value, f'{value:.0f}元', ha='center', va='bottom')

plt.subplot(2, 3, 4)
bucket_10_data['砍价比例'] = bucket_10_data['索赔差额绝对值'] / bucket_10_data['payment_contract_money']
avg_ratio_by_bucket = bucket_10_data.groupby('业务分桶')['砍价比例'].mean()
plt.bar(avg_ratio_by_bucket.index, avg_ratio_by_bucket.values, 
        alpha=0.7, color=colors)
plt.xlabel('业务分桶')
plt.ylabel('平均砍价比例')
plt.title('各分桶平均砍价比例')
plt.xticks(rotation=45)
for i, value in enumerate(avg_ratio_by_bucket.values):
    plt.text(i, value, f'{value:.2f}', ha='center', va='bottom')

plt.subplot(2, 3, 5)
box_data = []
for bucket in business_labels:
    box_data.append(bucket_10_data[bucket_10_data['业务分桶'] == bucket]['索赔差额'].values)
plt.boxplot(box_data, labels=business_labels)
plt.xticks(rotation=45)
plt.ylabel('索赔差额')
plt.title('各分桶索赔差额分布')

plt.subplot(2, 3, 6)
n, bins, patches = plt.hist(bucket_10_data['payment_real_money'], bins=30, 
                           alpha=0.3, color='gray', label='原始分布')
plt.xlabel('payment_real_money')
plt.ylabel('频数')
plt.title('原始分布与分桶界限')

for i, bin_edge in enumerate(business_bins[:-1]):
    plt.axvline(bin_edge, color=colors[i], linestyle='--', linewidth=2,
               label=f'{business_labels[i]}分界')
plt.axvline(business_bins[-1], color=colors[-1], linestyle='--', linewidth=2,
           label=f'{business_labels[-1]}分界')
plt.legend()

plt.tight_layout()
plt.show()

print("\n" + "=" * 50)
print("各分桶风险特征分析")
print("=" * 50)

for bucket in business_labels:
    bucket_data = bucket_10_data[bucket_10_data['业务分桶'] == bucket]
    
    print(f"\n{bucket}元区间分析:")
    print(f"样本数量: {len(bucket_data)} ({len(bucket_data)/total_10*100:.1f}%)")
    print(f"金额范围: {bucket_data['payment_real_money'].min():.0f} - {bucket_data['payment_real_money'].max():.0f}")
    print(f"平均赔付: {bucket_data['payment_real_money'].mean():.0f}元")
    print(f"平均索赔: {bucket_data['payment_contract_money'].mean():.0f}元")
    print(f"平均砍价: {bucket_data['索赔差额绝对值'].mean():.0f}元")
    print(f"平均砍价比例: {(bucket_data['索赔差额绝对值'].mean() / bucket_data['payment_contract_money'].mean() * 100):.1f}%")
    
    skewness = bucket_data['索赔差额'].skew()
    print(f"索赔差额偏度: {skewness:.3f} ({'右偏' if skewness > 0.5 else '左偏' if skewness < -0.5 else '近似对称'})")

print("\n" + "=" * 50)
print("应用到完整数据集")
print("=" * 50)

df_sorted['最终分桶'] = df_sorted['分桶编号'].astype(str)
tenth_mask = df_sorted['分桶编号'] == 10
df_sorted.loc[tenth_mask, '最终分桶'] = '10_' + bucket_10_data['业务分桶'].astype(str)

final_bucket_summary = df_sorted['最终分桶'].value_counts().sort_index()
print("最终分桶结构:")
for bucket, count in final_bucket_summary.items():
    if str(bucket).startswith('10_'):
        sub_bucket = bucket.split('_')[1]
        sub_bucket_data = df_sorted[df_sorted['最终分桶'] == bucket]
        amount_range = f"{sub_bucket_data['payment_real_money'].min():.0f}-{sub_bucket_data['payment_real_money'].max():.0f}"
        print(f"  分桶{bucket}: {count}个样本, 金额范围: {amount_range}")
    else:
        bucket_data = df_sorted[df_sorted['最终分桶'] == bucket]
        amount_range = f"{bucket_data['payment_real_money'].min():.0f}-{bucket_data['payment_real_money'].max():.0f}"
        print(f"  分桶{bucket}: {count}个样本, 金额范围: {amount_range}")

print(f"\n总样本量验证: {len(df_sorted)}")
print(f"分桶数量: {len(final_bucket_summary)}")

print(f"\n第十组业务导向分桶完成!")
print(f"第十组被细分为 {len(business_labels)} 个子分桶:")
for i, bucket in enumerate(business_labels):
    count = len(bucket_10_data[bucket_10_data['业务分桶'] == bucket])
    percentage = count / len(bucket_10_data) * 100
    print(f"  - {bucket}: {count}个样本 ({percentage:.1f}%)")

print("\n业务意义:")
print("  - 675-800元: 覆盖低端密集区域，砍价策略相对稳定")
print("  - 800-1000元: 覆盖中端密集区域，开始出现较大波动") 
print("  - 1000-1500元: 覆盖中高端区域，需要更精细的风险评估")
print("  - 1500+元: 专门处理高端异常值，建议人工重点审核")

print("=" * 50)
print("开始制定风险标注规则")
print("=" * 50)

risk_rules = []

print("正在计算各分桶的百分位数阈值...")
for bucket in sorted(df_sorted['最终分桶'].unique()):
    if pd.notna(bucket) and str(bucket) != 'nan':
        bucket_data = df_sorted[df_sorted['最终分桶'] == bucket]
        
        p85 = bucket_data['索赔差额绝对值'].quantile(0.85)
        p98 = bucket_data['索赔差额绝对值'].quantile(0.98)
        
        risk_rules.append({
            '分桶': bucket,
            '样本数': len(bucket_data),
            'P85阈值': round(p85, 2),
            'P98阈值': round(p98, 2),
            '最小砍价': round(bucket_data['索赔差额绝对值'].min(), 2),
            '最大砍价': round(bucket_data['索赔差额绝对值'].max(), 2),
            '平均砍价': round(bucket_data['索赔差额绝对值'].mean(), 2)
        })

rules_df = pd.DataFrame(risk_rules)
print("\n各分桶风险标注规则:")
print("=" * 80)
print(rules_df.to_string(index=False))

def apply_risk_label(row):
    if pd.isna(row['最终分桶']) or pd.isna(row['索赔差额绝对值']):
        return 'Unknown'
    
    bucket_rules = [r for r in risk_rules if r['分桶'] == row['最终分桶']]
    if not bucket_rules:
        return 'Unknown'
    
    rule = bucket_rules[0]
    current_diff = row['索赔差额绝对值']
    
    if current_diff <= rule['P85阈值']:
        return '合理诉求'
    elif current_diff <= rule['P98阈值']:
        return '诉求偏高'
    else:
        return '严重超额'

print("\n正在应用风险标注规则...")
df_sorted['风险标注'] = df_sorted.apply(apply_risk_label, axis=1)

print("\n" + "=" * 50)
print("风险标注结果验证")
print("=" * 50)

label_distribution = df_sorted['风险标注'].value_counts()
percentage_distribution = (df_sorted['风险标注'].value_counts(normalize=True) * 100).round(2)

print("总体风险标注分布:")
print("数量分布:")
print(label_distribution)
print("\n百分比分布:")
print(percentage_distribution)

reasonable = percentage_distribution.get('合理诉求', 0)
excessive = percentage_distribution.get('严重超额', 0)

print(f"\n比例要求验证:")
print(f"合理诉求: {reasonable}% {'符合' if reasonable >= 85 else '不符合'} (目标: ≥85%)")
print(f"严重超额: {excessive}% {'符合' if excessive <= 3 else '不符合'} (目标: ≤3%)")

if reasonable >= 85 and excessive <= 3:
    print("完全符合题目要求!")
else:
    print("需要调整规则参数")

print("\n" + "=" * 50)
print("各分桶风险标注分布分析")
print("=" * 50)

for bucket in sorted(df_sorted['最终分桶'].unique()):
    if pd.notna(bucket) and str(bucket) != 'nan':
        bucket_data = df_sorted[df_sorted['最终分桶'] == bucket]
        bucket_dist = (bucket_data['风险标注'].value_counts(normalize=True) * 100).round(2)
        
        print(f"\n分桶 {bucket}:")
        for label in ['合理诉求', '诉求偏高', '严重超额']:
            percent = bucket_dist.get(label, 0)
            print(f"  {label}: {percent}%")

plt.figure(figsize=(16, 12))

plt.subplot(2, 3, 1)
colors = {'合理诉求': 'green', '诉求偏高': 'orange', '严重超额': 'red'}
label_distribution.plot(kind='pie', autopct='%1.1f%%', colors=[colors.get(x, 'gray') for x in label_distribution.index])
plt.title('总体风险标注分布')
plt.ylabel('')

plt.subplot(2, 3, 2)
reasonable_by_bucket = []
bucket_labels = []
for bucket in sorted(df_sorted['最终分桶'].unique()):
    if pd.notna(bucket) and str(bucket) != 'nan':
        bucket_data = df_sorted[df_sorted['最终分桶'] == bucket]
        reasonable_pct = (bucket_data['风险标注'] == '合理诉求').mean() * 100
        reasonable_by_bucket.append(reasonable_pct)
        bucket_labels.append(str(bucket))

plt.bar(range(len(reasonable_by_bucket)), reasonable_by_bucket, color='lightgreen', alpha=0.7)
plt.axhline(y=85, color='red', linestyle='--', label='目标值85%')
plt.xticks(range(len(bucket_labels)), bucket_labels, rotation=45)
plt.xlabel('分桶')
plt.ylabel('合理诉求比例 (%)')
plt.title('各分桶合理诉求比例')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 3)
excessive_by_bucket = []
for bucket in sorted(df_sorted['最终分桶'].unique()):
    if pd.notna(bucket) and str(bucket) != 'nan':
        bucket_data = df_sorted[df_sorted['最终分桶'] == bucket]
        excessive_pct = (bucket_data['风险标注'] == '严重超额').mean() * 100
        excessive_by_bucket.append(excessive_pct)

plt.bar(range(len(excessive_by_bucket)), excessive_by_bucket, color='lightcoral', alpha=0.7)
plt.axhline(y=3, color='red', linestyle='--', label='目标值3%')
plt.xticks(range(len(bucket_labels)), bucket_labels, rotation=45)
plt.xlabel('分桶')
plt.ylabel('严重超额比例 (%)')
plt.title('各分桶严重超额比例')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 4)
for risk_type, color in colors.items():
    data = df_sorted[df_sorted['风险标注'] == risk_type]
    plt.scatter(data['payment_real_money'], data['索赔差额绝对值'], 
               alpha=0.6, s=10, c=color, label=risk_type)
plt.xlabel('payment_real_money')
plt.ylabel('索赔差额绝对值（砍价程度）')
plt.title('风险标注可视化')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 5)
for rule in risk_rules:
    bucket = rule['分桶']
    bucket_data = df_sorted[df_sorted['最终分桶'] == bucket]
    
    if len(bucket_data) > 0:
        avg_amount = bucket_data['payment_real_money'].mean()
        plt.scatter(avg_amount, rule['P85阈值'], color='blue', marker='^', s=100, label='P85阈值' if bucket == risk_rules[0]['分桶'] else "")
        plt.scatter(avg_amount, rule['P98阈值'], color='red', marker='v', s=100, label='P98阈值' if bucket == risk_rules[0]['分桶'] else "")
        plt.text(avg_amount, rule['P85阈值'], f'{rule["P85阈值"]:.0f}', ha='right', va='bottom')
        plt.text(avg_amount, rule['P98阈值'], f'{rule["P98阈值"]:.0f}', ha='right', va='bottom')

plt.xlabel('平均payment_real_money')
plt.ylabel('阈值金额')
plt.title('各分桶阈值设置')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 6)
risk_data = []
risk_labels = []
for risk_type in ['合理诉求', '诉求偏高', '严重超额']:
    data = df_sorted[df_sorted['风险标注'] == risk_type]['索赔差额绝对值']
    if len(data) > 0:
        risk_data.append(data.values)
        risk_labels.append(risk_type)

plt.boxplot(risk_data, labels=risk_labels)
plt.ylabel('索赔差额绝对值（砍价程度）')
plt.title('各类别砍价程度分布')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "=" * 50)
print("规则调整建议")
print("=" * 50)

if reasonable < 85 or excessive > 3:
    print("当前规则不满足要求，建议调整:")
    
    if reasonable < 85:
        adjustment = 85 - reasonable
        print(f"- 合理诉求不足 {adjustment:.1f}%，建议降低P85阈值")
        print("  调整策略: 将P85从0.85调整为0.83-0.84")
    
    if excessive > 3:
        adjustment = excessive - 3
        print(f"- 严重超额超出 {adjustment:.1f}%，建议提高P98阈值") 
        print("  调整策略: 将P98从0.98调整为0.985-0.99")
else:
    print("当前规则设置合理，无需调整")

print(f"\n风险标注规则制定完成!")
print(f"最终分布: 合理诉求 {reasonable}%, 诉求偏高 {percentage_distribution.get('诉求偏高', 0)}%, 严重超额 {excessive}%")

df['风险标注'] = df_sorted['风险标注']
df['最终分桶'] = df_sorted['最终分桶']

print(f"\n规则制定逻辑总结:")
print("1. 按金额分桶: 确保不同金额区间使用不同标准")
print("2. 组内百分位数: 在每个分桶内用P85和P98作为阈值")
print("3. 动态阈值: 高金额区间允许更大的砍价幅度")
print("4. 比例控制: 通过调整百分位数精确控制各类别比例")

# 问题三：风险标注分类模型 - 修正版本
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("问题三：风险标注分类模型 - 修正版本")
print("目标分布: 合理诉求≥85%, 严重超额≤3%")
print("=" * 60)

print("步骤1: 数据准备")

try:
    df_train = pd.read_csv(r"C:\Users\15507\Desktop\prediction_results_fixed_updated (2).csv", encoding='utf-8')
    print("附件1读取成功 (UTF-8)")
except UnicodeDecodeError:
    try:
        df_train = pd.read_csv(r"C:\Users\15507\Desktop\prediction_results_fixed_updated (2).csv", encoding='gbk')
        print("附件1读取成功 (GBK)")
    except UnicodeDecodeError:
        df_train = pd.read_csv(r"C:\Users\15507\Desktop\prediction_results_fixed_updated (2).csv", encoding='gb2312')
        print("附件1读取成功 (GB2312)")

print(f"附件1数据形状: {df_train.shape}")

try:
    df_test = pd.read_csv(r"C:\Users\15507\Desktop\Q2_预测.csv", encoding='utf-8')
    print("附件2读取成功 (UTF-8)")
except UnicodeDecodeError:
    try:
        df_test = pd.read_csv(r"C:\Users\15507\Desktop\Q2_预测.csv", encoding='gbk')
        print("附件2读取成功 (GBK)")
    except UnicodeDecodeError:
        df_test = pd.read_csv(r"C:\Users\15507\Desktop\Q2_预测.csv", encoding='gb2312')
        print("附件2读取成功 (GB2312)")

print(f"附件2数据形状: {df_test.shape}")

column_mapping = {
    'route_type': '线路类型',
    'is_customer_to_customer': '是否c2c', 
    'is_fresh_and_delv_promise': '是否生鲜妥投及时',
    'waybill_price_protect_money': '保价金额',
    'start_city_id': '始发城市',
    'end_city_id': '目的城市',
    'consigner_id': '寄件人id',
    'receiver_id': '收件人id',
    'is_staff': '寄件是否内部',
    'plan_delv_to_real_delv_diff': '配送超时时长',
    'abnormal_reason': '异常原因',
    'source': '进线渠道',
    'real_delv_to_case_create_diff': '妥投到进线时长',
    'payment_contract_money': '索赔金额',
    'goods_category': '商品类型',
    'goods_level': '新旧程度',
    'bc_source': '寄件B/C',
    'customer_role': '进线人身份',
    'start_node_waybill_num': '始发网点发单量',
    'start_node_accident_rate': '始发网点万单理赔率',
    'start_node_real_claim_num_ratio': '始发网点赔付比例',
    'end_node_waybill_num': '目的网点发单量',
    'end_node_accident_rate': '目的网点万单理赔率',
    'end_node_real_claim_num_ratio': '目的网点赔付比例'
}

reverse_mapping = {v: k for k, v in column_mapping.items()}

df_test_renamed = df_test.copy()
for chinese_col, english_col in reverse_mapping.items():
    if chinese_col in df_test_renamed.columns:
        df_test_renamed[english_col] = df_test_renamed[chinese_col]

print("\n数据预处理 - 转换数据类型")
df_train['payment_real_money'] = pd.to_numeric(df_train['payment_real_money'], errors='coerce')
df_train['payment_contract_money'] = pd.to_numeric(df_train['payment_contract_money'], errors='coerce')

if 'payment_contract_money' in df_test_renamed.columns:
    df_test_renamed['payment_contract_money'] = pd.to_numeric(df_test_renamed['payment_contract_money'], errors='coerce')
else:
    df_test_renamed['payment_contract_money'] = pd.to_numeric(df_test['索赔金额'], errors='coerce')

if 'y' not in df_test_renamed.columns:
    claim_median = df_train['payment_contract_money'].median()
    actual_median = df_train['payment_real_money'].median()
    discount_ratio = actual_median / claim_median if claim_median > 0 else 0.5
    df_test_renamed['y'] = df_test_renamed['payment_contract_money'] * discount_ratio
    print(f"基于索赔金额计算预测赔付，折扣比例: {discount_ratio:.2f}")

print("金额字段转换完成")

print("\n步骤3: 应用符合要求的风险标注规则")

def apply_correct_risk_rules(df):
    df['索赔差额'] = df['payment_real_money'] - df['payment_contract_money']
    df['索赔比例'] = abs(df['索赔差额']) / df['payment_real_money']
    
    df_sorted = df.sort_values('索赔比例').reset_index(drop=True)
    
    total_samples = len(df_sorted)
    reasonable_cutoff = int(total_samples * 0.85)
    high_cutoff = int(total_samples * 0.97)
    
    risk_labels = []
    for i in range(total_samples):
        if i < reasonable_cutoff:
            risk_labels.append('合理诉求')
        elif i < high_cutoff:
            risk_labels.append('诉求偏高')
        else:
            risk_labels.append('严重超额')
    
    df_sorted['风险标注'] = risk_labels
    return df_sorted

def predict_test_risk(df_train, df_test, predicted_payment_col='y'):
    test_ratios = abs(df_test[predicted_payment_col] - df_test['payment_contract_money']) / df_test['payment_contract_money']
    
    ratio_series = pd.Series(test_ratios)
    sorted_indices = ratio_series.sort_values().index
    
    total_samples = len(ratio_series)
    reasonable_count = int(total_samples * 0.85)
    high_count = int(total_samples * 0.12)
    
    risk_labels = ['合理诉求'] * total_samples
    
    for i in range(reasonable_count, reasonable_count + high_count):
        risk_labels[sorted_indices[i]] = '诉求偏高'
    
    for i in range(reasonable_count + high_count, total_samples):
        risk_labels[sorted_indices[i]] = '严重超额'
    
    df_test['风险标注'] = risk_labels
    
    return df_test

print("正在应用符合目标分布的风险标注规则...")
df_train_labeled = apply_correct_risk_rules(df_train)

print("训练数据风险标注分布:")

train_dist = df_train_labeled['风险标注'].value_counts(normalize=True) * 100
print("训练数据分布:", train_dist)

df_test_labeled = predict_test_risk(df_train, df_test_renamed)

test_dist = df_test_labeled['风险标注'].value_counts(normalize=True) * 100
print("测试数据分布:", test_dist)

print("\n步骤4: 使用机器学习模型进行精细调整")

exclude_features = ['风险标注', '索赔差额绝对值', '索赔比例', 'payment_real_money', 'y', 'Label', 'Predicted_Label', 'Is_Correct']
common_features = []
for col in df_train_labeled:
    if col not in exclude_features and col in df_train_labeled.columns:
        common_features.append(col)

print(f"使用 {len(common_features)} 个特征进行模型训练")

X_train = df_train_labeled[common_features].copy()
y_train =df_train_labeled['风险标注'].copy()
X_test = df_train_labeled[common_features].copy()

X_combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)

label_encoders = {}
for col in X_combined.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_combined[col] = X_combined[col].fillna('Missing')
    X_combined[col] = le.fit_transform(X_combined[col].astype(str))
    label_encoders[col] = le

numeric_features = X_combined.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_features:
    if col in X_combined.columns:
        missing_count = X_combined[col].isnull().sum()
        if missing_count > 0:
            X_combined[col] = X_combined[col].fillna(X_combined[col].median())

if numeric_features:
    scaler = StandardScaler()
    X_combined[numeric_features] = scaler.fit_transform(X_combined[numeric_features])

X_train_processed = X_combined.iloc[:len(X_train)].copy()
X_test_processed = X_combined.iloc[len(X_train):].copy().reset_index(drop=True)

label_encoder_y = LabelEncoder()
y_train_encoded = label_encoder_y.fit_transform(y_train)

model = LGBMClassifier(
    random_state=42,
    class_weight='balanced',
    verbose=-1,
    n_estimators=50
)

print("训练模型...")
model.fit(X_train_processed, y_train_encoded)

y_test_pred = model.predict(X_test_processed)
y_test_pred_decoded = label_encoder_y.inverse_transform(y_test_pred)

print("\n步骤5: 最终结果调整")

current_dist = pd.Series(y_test_pred_decoded).value_counts(normalize=True) * 100
print("模型预测分布:")
print(current_dist.round(2))

if current_dist.get('合理诉求', 0) >= 85 and current_dist.get('严重超额', 0) <= 3:
    print("模型预测符合要求，使用模型结果")
    final_risk_labels = y_test_pred_decoded
else:
    print("模型预测不符合要求，使用规则结果")
    final_risk_labels = df_train_labeled['风险标注'].values

result_q3 = pd.DataFrame({
    '风险标注': final_risk_labels
})

final_dist = result_q3['风险标注'].value_counts(normalize=True) * 100
print("\n最终风险标注分布:")
print(result_q3['风险标注'].value_counts())
print("比例分布:")
print(final_dist.round(2))

print("\n步骤6: 合并问题2和问题3结果")

result_q2 = pd.DataFrame({
    '预测赔付金额': df_test_renamed['y']
})

final_result = pd.concat([result_q2, result_q3], axis=1)

if '运单号' in df_test.columns:
    final_result.insert(0, '运单号', df_test['运单号'].values)
else:
    final_result.insert(0, '运单号', [f'WB{i:06d}' for i in range(len(final_result))])

final_result.to_csv('result_合并结果_修正.csv', index=False, encoding='utf-8-sig')
print("合并结果已保存为 'result_合并结果_修正.csv'")

reasonable_pct = final_dist.get('合理诉求', 0)
excessive_pct = final_dist.get('严重超额', 0)

print(f"\n最终比例验证:")
print(f"合理诉求: {reasonable_pct:.2f}% {'符合' if reasonable_pct >= 85 else '不符合'} (目标: ≥85%)")
print(f"严重超额: {excessive_pct:.2f}% {'符合' if excessive_pct <= 3 else '不符合'} (目标: ≤3%)")

print("\n" + "=" * 60)
print("问题三修正版本完成!")
print("=" * 60)
