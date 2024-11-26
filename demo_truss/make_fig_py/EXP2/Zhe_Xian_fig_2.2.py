import matplotlib.pyplot as plt
import pandas as pd
import os

# 获取当前脚本所在目录
dir_path = os.path.dirname(__file__)

# 读取CSV文件
df = pd.read_csv(os.path.join(dir_path, 'input', 'exp2.2.csv'), encoding='utf-8')

# 提取数据
x = df['N']
# WC,SO,WCP,WP,UK
columns_to_plot = ['CW', 'SO', 'WCP', 'WP','UK']

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
# styles = []
# colors = ['#87CEEB', 'Green', 'black', 'red', '#FFA500']
# colors = ['#808080', '#F94141', '#F3B169', '#589FF3', '#37AB78']
colors = ['dimgray', '#FF9900', '#F94141', '#37AB78', 'b']



# 创建图形和轴
fig, ax = plt.subplots(figsize=(6, 4))

# 循环绘制每条线
for col, marker, color in zip(columns_to_plot, markers, colors):
    ax.plot(x, df[col].astype(float), marker=marker,markersize=10, linestyle='-', color=color, label=col)

# 设置轴标签和标题
ax.set_yscale('log')
ax.set_ylabel('Time (seconds)',fontsize=20)
ax.set_xlabel('Value of $n_{cut}$',fontsize=23)  
ax.tick_params(axis='y', labelsize=18)
ax.set_xticks(x)
ax.set_xticklabels(x, fontsize=18)

# 添加网格线
ax.grid(True, linestyle='--', alpha=0.5)

# 添加图例
# ax.legend()
ax.legend(fontsize=16,ncol=3)


#  保存图
plt.savefig(dir_path+'/output/fig2.2.pdf', format='pdf', dpi=900, bbox_inches='tight')
# 显示图表
# plt.show()
plt.close(fig)