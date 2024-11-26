import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from brokenaxes import brokenaxes
from matplotlib.gridspec import GridSpec

dir_path = os.path.dirname(__file__)

# --------------CSV处理--------------
file_name = '1.1'
# 使用pandas读取CSV文件
df = pd.read_csv(dir_path + '/input/exp' + file_name + '.csv', encoding='utf-8')

# 提取数据
x = df['Name']
for_x_lable = ['CP', 'RG', 'M4', 'SS', 'WP']
columns_to_plot = ['WC', 'Ros', 'PKT', 'OPT-HPU', 'TDT']
colors = ['orange', 'lightBlue', 'mediumpurple', '#90EE90', 'red']
patterns = ['////', 'xx', '...', '\\\\\\', '|||||']

# --------------开始画图--------------
column_N = x.tolist()
x = np.arange(len(column_N))

# 创建图表和GridSpec
fig = plt.figure(figsize=(10, 3))
gs = GridSpec(1, 1, figure=fig)

# 使用brokenaxes创建带有断轴的图表
bax = brokenaxes(ylims=((0, 25), (35, 40)), hspace=0.05, subplot_spec=gs[0])

for i, (col, c_N, pattern, color) in enumerate(zip(columns_to_plot, columns_to_plot, patterns, colors)):
    bax.bar(x + (i - 1) * 0.15, df[col].astype(float), 0.15, label=c_N, color='white', hatch=pattern, edgecolor=color, lw=.6)

# 添加一些文本标签
bax.set_ylabel('Time (seconds)', fontsize=14)
bax.set_xticks(x)
bax.set_xticklabels(for_x_lable, fontsize=14)
bax.legend(fontsize=12, loc='upper left', ncol=3)

# 保存图
plt.savefig(dir_path + '/output/fig' + file_name + '.pdf', format='pdf', dpi=900, bbox_inches='tight')
# 显示图表（可选）
# plt.show()
plt.close(fig)