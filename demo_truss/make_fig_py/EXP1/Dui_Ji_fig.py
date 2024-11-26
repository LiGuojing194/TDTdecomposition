import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

dir_path = os.path.dirname(__file__)

# plt.rc('font', family='Times New Roman')

# --------------CSV处理--------------
# 使用pandas读取CSV文件
df = pd.read_csv(dir_path+'/input/exp1.5.csv', encoding='utf-8')
 
# 假设CSV文件的列名分别为'N', 'A', 'B'， 你可以直接通过列名来访问数据
column_N = df['Name'].tolist()
column_A = df['suptime'].astype(float).tolist()
# 列B应该是总时间与列A的差值
column_B = df['totaltime'].astype(float).tolist()


# --------------开始画图--------------
# 位置
x = np.arange(len(column_N))
 
# 创建堆叠柱状图
width = 0.35  # 柱子的宽度
fig, ax = plt.subplots()
# 为x轴标签; y轴值; 柱状图宽度; 图例标签; 颜色，hatch="//"设置填充图案，ec='k'设置边框颜色为黑色，lw=.6设置边框宽度
rects1 = ax.bar(x, column_A, width, label='ComputeSup time',color='orange',hatch="//",ec='k',lw=.6)
rects2 = ax.bar(x, column_B, width, bottom=column_A, label='Total time',color='#87CEEB',ec='k',lw=.6)
# 填充还有 "//" "..." "\\\\"
 
# 添加一些文本标签
ax.set_ylabel('Time (seconds)', fontsize=13)
# ax.set_ylim(0,12)   # 设置y轴范围
ax.set_title('Stacked Bar Chart')
ax.set_xticks(x)
ax.set_xticklabels(column_N, fontsize=13)

# 设置刻度线的样式，包括方向、大小、长度、宽度，以及是否显示顶部和右侧的刻度线。
# ax.tick_params(direction='out',labelsize=12,length=5.5,width=1,top=False,right=False)


ax.legend()  # 图例
# 添加图例，设置字体大小、是否显示图例框、图例位置以及列数。
# ax.legend(fontsize=11,frameon=False,loc='upper center',ncol=4)

# # 额外添加一些文本
# text_font = {'size':'16','weight':'bold','color':'black'}
# ax.text(.03,.93,"(a) ",transform = ax.transAxes,fontdict=text_font,zorder=4) 

#  保存图
plt.savefig(dir_path+'/output/fig1.5.pdf', format='pdf', dpi=900, bbox_inches='tight')
# 显示图表
# plt.show()
plt.close(fig)