import matplotlib.pyplot as plt
import pandas as pd
import os

# 获取当前脚本所在目录
dir_path = os.path.dirname(__file__)

# 读取CSV文件
df = pd.read_csv(os.path.join(dir_path, 'input', 'exp3.2.csv'), encoding='utf-8')

# 提取数据
x = df['N']
# TDT-origin	+Comp	+Update	 +both
columns_to_plot = ['TDT-origin', '+Comp', '+Update', '+both']

markers = ['o', '*', '^', 's']  # 可以根据需要调整标记
"""
s : 方块状
o : 实心圆
^ : 正三角形
v : 反正三角形
+ : 加好
* : 星号
x : x号
p : 五角星
1 : 三脚架标记
2 : 三脚架标记
"""
styles = ['--','-','-','-']
# colors = ['#87CEEB', 'Green', 'black', 'red']
colors = ['#808080', '#37AB78', '#589FF3', '#F94141']


# 创建图形和轴
fig, ax = plt.subplots(figsize=(10, 3))

# 循环绘制每条线
for col, marker,style, color in zip(columns_to_plot, markers,styles, colors):
    ax.plot(x, df[col].astype(float), marker=marker,markersize=10, linestyle=style, color=color, label=col)

# 设置轴标签和标题
ax.set_yscale('log')
ax.set_ylabel('Time (seconds)', fontsize=14)
ax.tick_params(axis='y', labelsize=14)

# ax.set_xlabel('Value of $n_{cut}$')  
ax.set_xticks(x)
ax.set_xticklabels(x, fontsize=14)

# 添加网格线
ax.grid(True, linestyle='--', alpha=0.5)

# 添加图例
# ax.legend()
ax.legend(fontsize=14,loc='upper left',ncol=2)

#  保存图
plt.savefig(dir_path+'/output/fig3.2.pdf', format='pdf', dpi=900, bbox_inches='tight')
# 显示图表
# plt.show()
plt.close(fig)