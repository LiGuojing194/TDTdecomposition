import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from brokenaxes import brokenaxes



dir_path = os.path.dirname(__file__)

# plt.rc('font', family='Times New Roman')

# --------------CSV处理--------------
file_name = '1.1'
# 使用pandas读取CSV文件
df = pd.read_csv(dir_path+'/input/exp'+file_name+'.csv', encoding='utf-8')
 
# # 假设CSV文件的列名分别为'N', 'A', 'B'， 你可以直接通过列名来访问数据
# column_N = df['N'].tolist()
# column_A = df['A'].astype(float).tolist()
# # 列B应该是总时间与列A的差值
# column_B = df['B'].astype(float).tolist()

# 提取数据
x = df['Name']
# WC	Ros	PKT	OPT-HPU	TDT
# for_x_lable = ['cit-Patents', 'rgg-n-2-24-s0', 'mawi4', 'soc-sinaweibo', 'wikipedia']
# CP RG M4 SS WP
for_x_lable = ['CP','RG','M4','SS','WP']
columns_to_plot = ['WC', 'Ros', 'PKT', 'OPT-HPU', 'TDT']
colors = ['orange', 'lightBlue',  'mediumpurple','#90EE90', 'red']  #CornflowerBlue
patterns = ['////', 'xx', '...','\\\\\\','|||||']



# --------------开始画图--------------
# 位置
column_N = x.tolist()
x = np.arange(len(column_N))



# 创建柱状图
width = 0.15  # 柱子的宽度

fig, ax = plt.subplots(figsize=(10, 3))
# 为x轴标签; y轴值; 柱状图宽度; 图例标签; 颜色，hatch="//"设置填充图案，ec='k'设置边框颜色为黑色，lw=.6设置边框宽度
# rects1 = ax.bar(x - width/2, column_A, width, label='Column A',color='#87CEEB',hatch="//",ec='k',lw=.6)
# rects2 = ax.bar(x + width/2, column_B, width, label='Column B',color='#FFFFE0',hatch="xx",ec='k',lw=.6)

# for col,c_N, pattern, color in zip(columns_to_plot, column_N, patterns, colors):
#     ax.bar(x - width/2, df[col].astype(float), width, label=c_N ,hatch=pattern ,ec=color,lw=.6)
for i, (col, c_N, pattern, color) in enumerate(zip(columns_to_plot, columns_to_plot, patterns, colors)):
    ax.bar(x + (i - 1) * width, df[col].astype(float), width, label=c_N,color='white', hatch=pattern, edgecolor=color, lw=.6)
 




# 浅蓝色（#87CEEB）、浅绿色（#90EE90）、浅黄色（#FFFFE0）
# 浅蓝色（#ADD8E6）,橙色（#FFA500）、浅灰色（#D3D3D3）,浅紫色（#E6E6FA）、淡黄色（#FFFACD）

# rects1 = ax.bar(x - width/2, column_A, width, label='Column A',color='lightBlue',hatch="//",ec='k',lw=.6)
# rects2 = ax.bar(x + width/2, column_B, width, label='Column B',color='mediumpurple',hatch="xx",ec='k',lw=.6)
# 填充还有 "//" "..." "\\\\"
 
# 添加一些文本标签
ax.set_ylabel('Time (seconds)', fontsize=14)

# ax.set_ylim(0,120)   # 设置y轴范围
# ax.set_title('Stacked Bar Chart')
ax.set_xticks(x)
# 斜着的x轴table
# ax.set_xticklabels(for_x_lable, rotation=45, ha='right', fontsize=13)
ax.set_xticklabels(for_x_lable, fontsize=14)


# 设置刻度线的样式，包括方向、大小、长度、宽度，以及是否显示顶部和右侧的刻度线。
# ax.tick_params(direction='out',labelsize=12,length=5.5,width=1,top=False,right=False)

ax.legend(fontsize=12,ncol=3)  # 图例
# # 添加图例，设置字体大小、是否显示图例框、图例位置以及列数。
# ax.legend(fontsize=11,frameon=False,loc='upper center',ncol=4)

# # 额外添加一些文本
# text_font = {'size':'16','weight':'bold','color':'black'}
# ax.text(.03,.93,"(a) ",transform = ax.transAxes,fontdict=text_font,zorder=4) 


# 加子图框
axins = inset_axes(ax, width="30%", height="20%", loc='lower left',
                   bbox_to_anchor=(0.08, 0.3, 1, 1),
                   bbox_transform=ax.transAxes)

width = 0.1
# 绘制柱状图
for i, (col, c_N, pattern, color) in enumerate(zip(columns_to_plot, column_N, patterns, colors)):
    # 只取每个col的第一组数据（在这个例子中，每组数据只有一个元素）
    data_value = df[col].iloc[0]
    axins.bar(x + (i - 1) * width, data_value, width, label=c_N, color='white', hatch=pattern, edgecolor=color, lw=.6)





# 调整子坐标系的显示范围
axins.set_xlim(-0.22, 0.4)
axins.set_ylim(0, 0.5)
axins.xaxis.set_visible(False)


# loc1 loc2: 坐标系的四个角
# 1 (右上) 2 (左上) 3(左下) 4(右下)
mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec='k', lw=1)






#  保存图
plt.savefig(dir_path+'/output/fig'+file_name+'.pdf', format='pdf', dpi=900, bbox_inches='tight')
# 显示图表
# plt.show()
plt.close(fig)