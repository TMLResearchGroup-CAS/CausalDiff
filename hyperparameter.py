import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.size'] = 41
plt.subplots(figsize=(20, 8))


x = [0,0.1,0.2,0.4,0.6,0.8] #点的横坐标
y1 = [98.9,98.2,97.8,96.1,92.4,86.6] #线1的纵坐标
y2 = [63.3,62.7,62.4,58.0,48.0,30.0] #线2的纵坐标
y3 = [61.0,60.4,60.1,52.1,41.2,22.8] #线2的纵坐标
y4 = [18.4,18.0,17.8,16.0,12.8,5.2] #线2的纵坐标


plt.axhline(y=86.6, color='0.7', linestyle='--')
plt.axhline(y=30.0, color='0.7', linestyle='--')
plt.axhline(y=22.8, color='0.7', linestyle='--')
plt.axhline(y=5.2, color='0.7', linestyle='--')

# plt.plot(x,y1,'o-',color = '#A95450',label="ASR", ms=15, linewidth=5.0) #o-:圆形
# plt.plot(x,y2,'s-',color = '#d85800',label="Boost", ms=15, linewidth=5.0) #s-:方形
# plt.plot(x,y3,'D-',color = '#73A366',label="T10R", ms=15, linewidth=5.0) #o-:圆形
# plt.plot(x,y4,'p-',color = '#5C72AF',label="T5R", ms=15, linewidth=5.0) #o-:圆形


# plt.plot(x,y1,'o-',color = '#C07A92',label="ASR", ms=15, linewidth=5.0) #o-:圆形
# plt.plot(x,y2,'s-',color = '#DFC286',label="Boost", ms=15, linewidth=5.0) #s-:方形
# plt.plot(x,y3,'D-',color = '#80AFBF',label="T10R", ms=15, linewidth=5.0) #o-:圆形
# plt.plot(x,y4,'p-',color = '#608595',label="T5R", ms=17, linewidth=5.0) #o-:圆形


# plt.plot(x,y1,'o-',color = '#db7e6a',label="ASR", ms=15, linewidth=5.0) #o-:圆形
# plt.plot(x,y2,'s-',color = '#79ced1',label="Boost", ms=15, linewidth=5.0) #s-:方形
# plt.plot(x,y3,'D-',color = '#007d7d',label="T10R", ms=15, linewidth=5.0) #o-:圆形
# plt.plot(x,y4,'p-',color = '#f58422',label="T5R", ms=17, linewidth=5.0) #o-:圆形


plt.plot(x,y1,'o-',color = '#c5715f',label="ASR", ms=15, linewidth=5.0) #o-:圆形
plt.plot(x,y2,'s-',color = '#6cb9bc',label="Boost", ms=15, linewidth=5.0) #s-:方形
plt.plot(x,y3,'D-',color = '#007070',label="T10R", ms=15, linewidth=5.0) #o-:圆形
plt.plot(x,y4,'p-',color = '#dc761e',label="T5R", ms=17, linewidth=5.0) #o-:圆形


plt.xticks(np.arange(0, 0.9, step=0.1))

plt.text(-0.08, -1, '$\\beta$', fontsize=41, ha='left', va='top')

plt.legend(loc='center left', bbox_to_anchor=(0.99, 0.29))

# plt.legend(loc = "best") #图例

plt.tight_layout()
plt.savefig('beta.pdf',format='pdf',bbox_inches='tight')
