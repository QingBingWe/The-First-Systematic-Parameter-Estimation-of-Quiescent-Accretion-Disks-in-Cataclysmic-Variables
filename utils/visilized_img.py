import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv("../parameter_sdss_and_lamost.csv")
# 模拟两组黑体温度数据

sdss = data[data['Source']=='SDSS']['Outer Radius (lg(cm))']
print(sdss.describe())
lamost = data[data['Source']=='LAMOST']['Outer Radius (lg(cm))']# LAMOST 数据
print(lamost.describe())

plt.figure(figsize=(18,5))

# -------------------------------
# 1. 直方图
# -------------------------------

plt.subplot(1,3,1)
# LAMOST
plt.hist(lamost, bins=20,
         alpha=0.3,
         label="CV_LAMOST",
         color="skyblue",
         edgecolor="lightblue",   # 只画外边框
         linewidth=1.2,
         histtype="stepfilled")

# SDSS
plt.hist(sdss, bins=20,
         alpha=0.3,
         label="CV_SDSS",
         color="lightpink",
         edgecolor="orange",   # 只画外边框
         linewidth=1.2,
         histtype="stepfilled")


plt.xlabel("Outer Radius")
plt.ylabel("Number of CVs")
plt.legend()

# -------------------------------
# 2. 箱线图
# -------------------------------
plt.subplot(1,3,2)
plt.boxplot([sdss, lamost], labels=["SDSS", "LAMOST"], patch_artist=False)
plt.ylabel("Outer Radius")

# -------------------------------
# 3. 雷达图
# -------------------------------
plt.subplot(1,3,3, polar=True)

# 将温度分区间，统计每区间个数
bins = np.linspace(7, 16, 10)
sdss_hist, _ = np.histogram(sdss, bins=bins)
lamost_hist, _ = np.histogram(lamost, bins=bins)

# 雷达图坐标
labels = [f"{int(b)}" for b in bins[:-1]]
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]  # 闭合

# 数据闭合
sdss_data = sdss_hist.tolist() + sdss_hist[:1].tolist()
lamost_data = lamost_hist.tolist() + lamost_hist[:1].tolist()

# 绘制 SDSS
plt.polar(angles, sdss_data, color="purple", linewidth=0, label="SDSS")
plt.fill(angles, sdss_data, color="purple", alpha=0.3)

# 绘制 LAMOST
plt.polar(angles, lamost_data, color="indianred", linewidth=0, label="LAMOST")
plt.fill(angles, lamost_data, color="indianred", alpha=0.3)

# 设置标签
plt.xticks(angles[:-1], labels)
plt.title("Outer Radius", y=1.1)
# plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
plt.legend()
# plt.yticks([])  # 去掉半径刻度
plt.gca().set_yticklabels([])  # 去掉半径文字


plt.tight_layout()
plt.savefig("Outer_Radius.png")
plt.show()
