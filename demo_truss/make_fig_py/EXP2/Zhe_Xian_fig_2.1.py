import matplotlib.pyplot as plt
import pandas as pd
import os

# 获取当前脚本所在目录
dir_path = os.path.dirname(__file__)

# 读取CSV文件
df = pd.read_csv(os.path.join(dir_path, 'input', 'exp2.1.csv'), encoding='utf-8')

# 提取数据
x = df['N']
# columns_to_plot = ['Al1', 'Al2', 'Al3', 'Al4', 'Al5']
# RN	RU	M4	P4	A0
columns_to_plot = ['RN', 'RU', 'M4', 'P4','A0']

markers = ['o', '*', '^', 's', 'v']  # 可以根据需要调整标记
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
# styles = []  3366CC
colors = ['b', '#FF9900', '#37AB78', '#F94141', 'dimgray']
# colors = ['#A5AEB7', '#7AB656', '#7E99F4', '#CC7C71', '#925EB0']
# colors = ['#808080', '#F94141', '#F3B169', '#589FF3', '#37AB78']



# 创建图形和轴
fig, ax = plt.subplots(figsize=(6, 4))

# 循环绘制每条线
for col, marker, color in zip(columns_to_plot, markers, colors):
    ax.plot(x, df[col].astype(float), marker=marker,markersize=10, linestyle='-', color=color, label=col)

# 设置轴标签和标题
ax.set_ylabel('Time (seconds)',fontsize=20)
ax.set_xlabel('Value of $n_{cut}$',fontsize=23)  
ax.tick_params(axis='y', labelsize=18)
ax.set_xticks(x)
ax.set_xticklabels(x, fontsize=18)

# 添加网格线
ax.grid(True, linestyle='--', alpha=0.5)

# 添加图例
# ax.legend()
ax.legend(fontsize=16,loc='upper right',ncol=3 )

#  保存图
plt.savefig(dir_path+'/output/fig2.1.pdf', format='pdf', dpi=900, bbox_inches='tight')
# 显示图表
# plt.show()
plt.close(fig)