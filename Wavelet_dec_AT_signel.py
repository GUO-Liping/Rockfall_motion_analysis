# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 11:27:58 2020

@author: zh'h
"""
from numpy import *
import numpy as np  
from numpy import genfromtxt 
import math 
from scipy import optimize, special
from pylab import *
import matplotlib.pyplot as plt
import pywt
from scipy import signal
import mpl_toolkits.mplot3d
import xlwt
from scipy import signal
'''
LparaPath = r"E:\学习资料\博士材料\AT_tr_shortcut_pro\python_Data\fau\Lpara.csv"   
Lpara = genfromtxt(LparaPath, delimiter=',')
wPath_AT = r"E:\学习资料\博士材料\AT_tr_shortcut_pro\python_Data\fau\tr_fau.csv"   
wpara_AT = genfromtxt(wPath_AT, delimiter=',')
t=np.arange(0, 0.00005, 0.000001)
t=t[0:50]

#电流变换阵
Q=np.mat([[2.419774138852552E-01,6.700695140292748E-01,-7.002179225265957E-01],[9.647496604840681E-01,-5.395036135003030E-01, 1.366345166620881E-01],[ 1.034650847642780E-01, 5.098457584296328E-01 ,7.007323810336268E-01]])
#电压变换阵
S=(Q.I).T


I_i1a2a=wpara_AT[0]                                    #入射接触线电流
I_i1b2b=wpara_AT[1]                                    #入射钢轨电流
I_i1c2c=wpara_AT[2]                                   #入射正馈线电流



vFault_1=np.array([I_i1a2a,I_i1b2b,I_i1c2c])
vFault_1S=Q.I*vFault_1

I12_i1a2a=vFault_1S[0]                             #解耦后0模
I12_i1b2b=vFault_1S[1]                             #解耦后线模1
I12_i1c2c=vFault_1S[2]                             #解耦后线模2
I12_i1a2a=np.array(I12_i1a2a.T)
I12_i1b2b=np.array(I12_i1b2b.T)
I12_i1c2c=np.array(I12_i1c2c.T)
'''
'''
小波包分解：
   小波包是为了克服小波分解在高频段的频率分辨率较差，而在低频段的时间分辨率较差
的问题的基础上而提出的。
它是一种更精细的信号分析的方法，提高了信号的时域分辨率。

能量谱：
   基于小波包分解提取多尺度空间能量特征的原理是把不同分解尺度上的信号能量求解出来，
将这些能量值按尺度顺序排列成特征向量供识别使用。
'''

'''
     是不是小波以一个尺度分解一次就是小波进行一层的分解？
     比如：[C,L]=wavedec(X,N,'wname')中，N为尺度，若为1，就是进行单尺度分解，也就是分解一层。   但是W=CWT(X,[2:2:128],'wname','plot')的分解尺度又是从2～128以2为步进的，这里的“分解尺度”跟上面那个“尺度”的意思一样吗？
    [C,L]=wavedec(X,N,'wname')中的N为分解层数, 不是尺度,'以wname'是DB小波为例, 如DB4, 4为消失矩,则一般滤波器长度为8, 阶数为7.

    wavedec针对于离散,CWT是连续的。

      多尺度又是怎么理解的呢？
     多尺度的理解: 如将0-pi定义为空间V0, 经过一级分解之后V0被分成0-pi/2的低频子空间V1和pi/2-pi的高频子空间W1, 然后一直分下去....得到 VJ+WJ+....W2+W1.   因为VJ和WJ是正交的空间, 且各W子空间也是相互正交的. 所以分解得到了是相互不包含的多个频域区间,这就是多分辩率分析, 即多尺度分析.
     当然多分辨率分析是有严格数学定义的,但完全可以从数字滤波器角度理解它.当然,你的泛函学的不错,也可以从函数空间角度理解.


      是不是说分解到W3、W2、W1、V3就是三尺度分解？
      简单的说尺度就是频率，不过是反比的关系．确定尺度关键还要考虑你要分析信号的采样频率大小，因为根据采样频率大小才能确定你的分析频率是多少．（采样定理）．然后再确定你到底分多少层．


      假如我这有一个10hz和50hz的正弦混合信号，采样频率是500hz，是不是就可以推断出10hz和50hz各自对应的尺度了呢？我的意思是，是不是有一个频率和尺度的换算公式？
     实际频率＝小波中心频率×采样频率/尺度


        在小波分解中，若将信号中的最高频率成分看作是1，则各层小波小波分解便是带通或低通滤波器，且各层所占的具体频带为（三层分解）a1:0~0.5 d1: 0.5~1; a2:0~0.25 d2: 0.25~0.5; a3: 0~0.125; d3:0.125~0.25   可以这样理解吗？如果我要得到频率为0.125~0.25的信号信息，是不是直接对d3的分解系数直接重构之后就是时域信息了？这样感觉把多层分解纯粹当作滤波器来用了，又怎么是多分辨分析？？ 怎样把时频信息同时表达出来？？
      这个问题非常好，我刚开始的时候也是被这个问题困惑住了，咱们确实是把它当成了滤波器来用了，也就是说我们只看重了小波分析的频域局部化的特性。但是很多人都忽略其时域局部化特性，因为小波是变时频分析的方法，根据测不准原理如果带宽大，则时窗宽度就要小。那么也就意味着如果我们要利用其时域局部化特性就得在时宽小的分解层数下研究，也就是低尺度下。这样我们就可以更容易看出信号在该段时间内的细微变化，但是就产生一个问题，这一段的频率带很宽，频率局部化就体现不出来了。
      对d3进行单支重构就可以得到0.125－0.25的信号了，当然频域信息可能保存的比较好，
但如果小波基不是对称的话，其相位信息会失真。


     小波变换主要也是用在高频特征提取上。

    层数不是尺度，小波包分解中，N应该是层数，个人理解对应尺度应该是2^N

    小波分解的尺度为a，分解层次为j。 如果是连续小波分解尺度即为a。离散小波分解尺度严格意义上来说为a＝2^j,在很多书上就直接将j称为尺度，因为一个j就对应者一个尺度a。其实两者是统一的。

'''
#wp = pywt.WaveletPacket(data=test_data, wavelet='db1',mode='symmetric',maxlevel=3)
#I12_i1b2b=I12_i1b2b[:,0]




def wpd_plt(wave,n):#  信号按照尺度分解的小波包系数图
    #wpd分解
    wp = pywt.WaveletPacket(data=wave, wavelet='db1',mode='symmetric',maxlevel=n)
 
    #计算每一个节点的系数，存在map中，key为'aa'等，value为列表
    map = {}
    map[1] = wave
    for row in range(1,n+1):
        for i in [node.path for node in wp.get_level(row, 'freq')]:  #i的取值依次为每层的节点
            map[i] = wp[i].data
            print(i)
    #作图
    plt.figure(figsize=(15, 10))
    plt.subplot(n+1,1,1) #绘制第一个图
    plt.plot(map[1])
    for i in range(2,n+2):
        level_num = pow(2,i-1)  #从第二行图开始，计算上一行图的2的幂次方
        #获取每一层分解的node：比如第三层['aaa', 'aad', 'add', 'ada', 'dda', 'ddd', 'dad', 'daa']
        re = [node.path for node in wp.get_level(i-1, 'freq')]  
        for j in range(1,level_num+1):
            plt.subplot(n+1,level_num,level_num*(i-1)+j)
            plt.plot(map[re[j-1]]) #列表从0开始
  




'''
wp=pywt.WaveletPacket(data=I_i1a2a,wavelet='db1',mode='symmetric')
获得小波包最大分解层数，maxlevel不设定数值则没有定义分解层数或者设定maxlevel=3，会根据data长度自动计算，返回的wp为小波包树
'''

'''
获得小波包树子节点,按频带频率进行排序
'''
#print([node.path for node in wp.get_level(1, 'freq')])

'''
打印小波家族
'''
pywt.families()

'''
小波包分解，提取系数
'''
#aaa=wp['a'].data
#aaa=np.nan_to_num(aaa)

'''
def wave_ener_spect(wave,n):#小波包能量pu  特征柱形图
    wp = pywt.WaveletPacket(data=wave, wavelet='db1',mode='symmetric',maxlevel=n)
    re = []  #第n层所有节点的分解系数
    for i in [node.path for node in wp.get_level(n, 'freq')]:
        re.append(wp[i].data)
        print(i)
        #第n层能量特征
    energy = []
    for i in re:
        energy.append(pow(np.linalg.norm(i,ord=None),2))
        
    plt.figure(figsize=(10, 7), dpi=80)
    N = np.power(2,n)
    energy[0]= 0
    values = energy
    index = np.arange(N)
    width = 0.45
    p2 = plt.bar(index, values, width, label="num", color="#87CEFA")
    plt.xlabel('clusters')
# 设置纵轴标签
    plt.ylabel('number of reviews')
# 添加标题
    plt.title('Cluster Distribution')
# 添加纵横轴的刻度
    plt.xticks(index, index)
# plt.yticks(np.arange(0, 10000, 10))
# 添加图例
    plt.legend(loc="upper right")
    plt.show()
    return energy
    



def wave_ener_entropy(wave,n):#小波能量熵
    wp = pywt.WaveletPacket(data=wave, wavelet='db1',mode='symmetric',maxlevel=n)
    N = np.power(2,n)
    m= np.ones(N)
    j=0
    re = []  #第n层所有节点的分解系数
    for i in [node.path for node in wp.get_level(n, 'freq')]:
        re.append(wp[i].data)
        m[j]=np.shape(wp[i].data)[0]
        j=j+1
        print(i)
        #第n层能量特征
    energy = []
    for i in re:
        energy.append(pow(np.linalg.norm(i,ord=None),2))
        energy_Norm=energy/np.sum(energy)
        print(i)
   
    # 再创建一个规格为 1 x 1 的子图
    # plt.subplot(1, 1, 1)
    # 柱子总数
    
    values = energy_Norm
    
    #values[0]= 0
    m=m.astype(np.int16)
    sum_nm=np.ones(N)
    for i in range(N):
        su=np.ones(m[i])
        for j in range(m[i]):
            su[j]=-energy_Norm[i]*(math.log(energy_Norm[i]))
            sum_nm[i]=sum(su)
    plt.plot(sum_nm,color="green")
    plt.show()
    # 包含每个柱子下标的序列
     # 创建一个点数为 8 x 6 的窗口, 并设置分辨率为 80像素/每英寸
    plt.figure(figsize=(10, 7), dpi=80)
    index = np.arange(N)
    # 柱子的宽度
    width = 0.45
    # 绘制柱状图, 每根柱子的颜色为紫罗兰色
    p2 = plt.bar(index, values, width, label="num", color="#87CEFA")
    # 设置横轴标签
    plt.xlabel('clusters')
    # 设置纵轴标签
    plt.ylabel('number of reviews')
    # 添加标题
    plt.title('Cluster Distribution')
    # 添加纵横轴的刻度
    #plt.xticks(index, ('7', '8', '9', '10', '11', '12', '13', '14'))
    plt.xticks(index, index)
    # plt.yticks(np.arange(0, 10000, 10))
    # 添加图例
    plt.legend(loc="upper right")
    plt.show()
    return sum_nm



小波包能量谱

n = 3
re = []  #第n层所有节点的分解系数
for i in [node.path for node in wp.get_level(n, 'freq')]:
    wp[i].data=np.nan_to_num(wp[i].data) #数据中存在nan，将nan转换为0，求范数时不影响最终值
    re.append(wp[i].data)
#第n层能量特征
energy = []
for i in re:
    energy.append(pow(np.linalg.norm(i,ord=None),2))
#for i in energy:
#    print(i)


绘制小波能量特征柱形图，节点顺序为频率从低到高

# 创建一个点数为 8 x 6 的窗口, 并设置分辨率为 80像素/每英寸
plt.figure(figsize=(10, 7), dpi=80)
# 再创建一个规格为 1 x 1 的子图
# plt.subplot(1, 1, 1)
# 柱子总数
N = 8
energy[0]= 0
values = energy
# 包含每个柱子下标的序列
index = np.arange(N)
# 柱子的宽度
width = 0.45
# 绘制柱状图, 每根柱子的颜色为紫罗兰色
p2 = plt.bar(index, values, width, label="num", color="#87CEFA")
# 设置横轴标签
plt.xlabel('clusters')
# 设置纵轴标签
plt.ylabel('number of reviews')
# 添加标题
plt.title('Cluster Distribution')
# 添加纵横轴的刻度
plt.xticks(index, ('1', '2', '3', '4', '5', '6', '7', '8'))
# plt.yticks(np.arange(0, 10000, 10))
# 添加图例
plt.legend(loc="upper right")
plt.show()
'''
'''
小波包系数重构测试
wp = pywt.WaveletPacket(data=test_data, wavelet='db1',mode='symmetric',maxlevel=3)
new_wp= pywt.WaveletPacket(data=None, wavelet='db1',mode='symmetric',maxlevel=3)

new_wp['da']=wp['da'].data

plt.plot(new_wp['da'].data)
plt.show()
plt.plot(new_wp.reconstruct(update=True))
plt.show()
print(new_wp.data)
plt.plot(new_wp['aa'].reconstruct(update=True))
plt.show()
plt.plot(new_wp['da'].reconstruct(update=True))
plt.show()
plt.plot(new_wp['daa'].reconstruct(update=True))
plt.show()
print([n.path for n in new_wp.get_leaf_nodes(False)])
'''

'''
逻辑回归二分类算法

def my_logistic(xx):
    return 1/(1+np.exp(-xx))

theta=np.ones(n) 
J0=np.ones(m)
J1=np.ones(m)
J2=np.ones(m)
J3=np.ones(m)
J4=np.ones(m)
J5=np.ones(m)
J6=np.ones(m)
J7=np.ones(m)
J8=np.ones(m)



for j in range(30000):
    for i in range(m):
        J0[i]=(y[i]-my_logistic(np.dot(feature[i,:],theta)))*feature[i,0]
        J1[i]=(y[i]-my_logistic(np.dot(feature[i,:],theta)))*feature[i,1]
        J2[i]=(y[i]-my_logistic(np.dot(feature[i,:],theta)))*feature[i,2]
        J3[i]=(y[i]-my_logistic(np.dot(feature[i,:],theta)))*feature[i,3]
        J4[i]=(y[i]-my_logistic(np.dot(feature[i,:],theta)))*feature[i,4]
        J5[i]=(y[i]-my_logistic(np.dot(feature[i,:],theta)))*feature[i,5]
        J6[i]=(y[i]-my_logistic(np.dot(feature[i,:],theta)))*feature[i,6]
        J7[i]=(y[i]-my_logistic(np.dot(feature[i,:],theta)))*feature[i,7]
        J8[i]=(y[i]-my_logistic(np.dot(feature[i,:],theta)))*feature[i,8]
    
    a=0.001
    theta[0]=theta[0]+a*np.sum(J0)
    theta[1]=theta[1]+a*np.sum(J1)
    theta[2]=theta[2]+a*np.sum(J2)
    theta[3]=theta[3]+a*np.sum(J3)
    theta[4]=theta[4]+a*np.sum(J4)
    theta[5]=theta[5]+a*np.sum(J5)
    theta[6]=theta[6]+a*np.sum(J6)
    theta[7]=theta[7]+a*np.sum(J7)
    theta[8]=theta[8]+a*np.sum(J8)

print (theta)

'''

def wave_ener_spect(wave,n):#小波包能量pu  特征柱形图
    wp = pywt.WaveletPacket(data=wave, wavelet='db1',mode='symmetric',maxlevel=n)
    re = []  #第n层所有节点的分解系数
    for i in [node.path for node in wp.get_level(n, 'freq')]:
        re.append(wp[i].data)
        print(i)
        #第n层能量特征
    energy = []
    for i in re:
        energy.append(pow(np.linalg.norm(i,ord=None),2))
        
    energy[0]= 0.0001  
    energy_Norm=energy/np.sum(energy)

    return energy_Norm
    


def wave_ener_entropy(wave,n):#小波能量熵
    wp = pywt.WaveletPacket(data=wave, wavelet='db1',mode='symmetric',maxlevel=n)
    N = np.power(2,n)
    m= np.ones(N)
    j=0
    re = []  #第n层所有节点的分解系数
    for i in [node.path for node in wp.get_level(n, 'freq')]:
        re.append(wp[i].data)
        m[j]=np.shape(wp[i].data)[0]
        j=j+1
        #第n层能量特征
    energy = []
    for i in re:
        energy.append(pow(np.linalg.norm(i,ord=None),2))
        
    energy[0]= 0.0001    
    energy_Norm=energy/np.sum(energy)
    m=m.astype(np.int16)
    sum_nm=np.ones(N)
    for i in range(N):
        su=np.ones(m[i])
        for j in range(m[i]):
            su[j]=-energy_Norm[i]*(math.log(energy_Norm[i]))
            sum_nm[i]=sum(su)
    return sum_nm
         
      
#wp = pywt.WaveletPacket(data=test_data, wavelet='db1',mode='symmetric',maxlevel=6)
#new_wp = pywt.WaveletPacket(data=None, wavelet='db1',mode='symmetric',maxlevel=3)

#wpd_plt(test_data,5)
#wave_ener_spect(test_data,5)
#wave_ener_entropy(test_data,5)

'''
ab=new_wp['daa']=wp['daa']
abv=ab.data
acc=new_wp.reconstruct(update=True)
plt.plot(abv,color="red")
plt.show()
plt.plot(acc,color="green")
plt.show()

m=[]
m=np.shape(new_wp['aaa'].data)[0]


print(m)
qwe=pow(np.linalg.norm(abv,ord=None),2)
qws=pow(np.linalg.norm(acc,ord=None),2)
print(qwe)
print(qws)
'''

'''
牵引变电所电压数据导入  导入的数据为接触线、钢轨、正馈线
'''

tr_tran_path = r"E:\学习资料\博士材料\AT_tr_shortcut_pro\python_Data\work\U_traction_data\tr_U_traction_data.csv"   
tr_tran_data = genfromtxt(tr_tran_path, delimiter=',')
tr_tran_x,tr_tran_y=np.shape(tr_tran_data)
tr_U_tran_C=range(0, tr_tran_x, 3)
tr_c_tran_data = tr_tran_data[tr_U_tran_C,:]
tr_C_tran_data=tr_c_tran_data[:,0:8192]

tr_U_tran_F=range(2, tr_tran_x, 3)
tr_f_tran_data = tr_tran_data[tr_U_tran_F,:]
tr_F_tran_data=tr_f_tran_data[:,0:8192]

lj_tran_path = r"E:\学习资料\博士材料\AT_tr_shortcut_pro\python_Data\work\U_traction_data\lj_U_traction_data.csv"   
lj_tran_data = genfromtxt(lj_tran_path, delimiter=',')
lj_tran_x,lj_tran_y=np.shape(lj_tran_data)
lj_U_tran_C=range(0, lj_tran_x, 3)
lj_c_tran_data = lj_tran_data[lj_U_tran_C,:]
lj_C_tran_data=lj_c_tran_data[:,0:8192]

lj_U_tran_F=range(2, lj_tran_x, 3)
lj_f_tran_data = lj_tran_data[lj_U_tran_F,:]
lj_F_tran_data=lj_f_tran_data[:,0:8192]

arc_tran_path = r"E:\学习资料\博士材料\AT_tr_shortcut_pro\python_Data\work\U_traction_data\arc_U_traction_data.csv"   
arc_tran_data = genfromtxt(arc_tran_path, delimiter=',')
arc_tran_x,arc_tran_y=np.shape(arc_tran_data)
arc_U_tran_C=range(0, arc_tran_x, 3)
arc_c_tran_data = arc_tran_data[arc_U_tran_C,:]
arc_C_tran_data=arc_c_tran_data[:,0:8192]

arc_U_tran_F=range(2, arc_tran_x, 3)
arc_f_tran_data = arc_tran_data[arc_U_tran_F,:]
arc_F_tran_data=arc_f_tran_data[:,0:8192]
'''
AT1电流数据导入  导入的数据为接触线、钢轨、正馈线
'''


tr_I_AT1_path = r"E:\学习资料\博士材料\AT_tr_shortcut_pro\python_Data\work\I_AT1_data\tr_I_AT1_data.csv"   
tr_I_AT1_data = genfromtxt(tr_I_AT1_path, delimiter=',')
tr_I_AT1_x,tr_I_AT1_y=np.shape(tr_I_AT1_data)
tr_I_AT1_C=range(0, tr_I_AT1_x, 3)
tr_I_c_AT1_data = tr_I_AT1_data[tr_I_AT1_C,:]
tr_I_C_AT1_data=tr_I_c_AT1_data[:,0:8192]

tr_I_AT1_F=range(2, tr_I_AT1_x, 3)
tr_I_f_AT1_data = tr_I_AT1_data[tr_I_AT1_F,:]
tr_I_F_AT1_data=tr_I_f_AT1_data[:,0:8192]


lj_I_AT1_path = r"E:\学习资料\博士材料\AT_tr_shortcut_pro\python_Data\work\I_AT1_data\lj_I_AT1_data.csv"   
lj_I_AT1_data = genfromtxt(lj_I_AT1_path, delimiter=',')
lj_I_AT1_x,lj_I_AT1_y=np.shape(lj_I_AT1_data)
lj_I_AT1_C=range(0, lj_I_AT1_x, 3)
lj_I_c_AT1_data = lj_I_AT1_data[lj_I_AT1_C,:]
lj_I_C_AT1_data=lj_I_c_AT1_data[:,0:8192]

lj_I_AT1_F=range(2, lj_I_AT1_x, 3)
lj_I_f_AT1_data = lj_I_AT1_data[lj_I_AT1_F,:]
lj_I_F_AT1_data=lj_I_f_AT1_data[:,0:8192]


arc_I_AT1_path = r"E:\学习资料\博士材料\AT_tr_shortcut_pro\python_Data\work\I_AT1_data\arc_I_AT1_data.csv"   
arc_I_AT1_data = genfromtxt(arc_I_AT1_path, delimiter=',')
arc_I_AT1_x,arc_I_AT1_y=np.shape(arc_I_AT1_data)
arc_I_AT1_C=range(0, arc_I_AT1_x, 3)
arc_I_c_AT1_data = arc_I_AT1_data[arc_I_AT1_C,:]
arc_I_C_AT1_data=arc_I_c_AT1_data[:,0:8192]

arc_I_AT1_F=range(2, arc_I_AT1_x, 3)
arc_I_f_AT1_data = arc_I_AT1_data[arc_I_AT1_F,:]
arc_I_F_AT1_data=arc_I_f_AT1_data[:,0:8192]

'''
AT3电流数据导入  导入的数据为接触线、钢轨、正馈线
'''

tr_I_AT3_path = r"E:\学习资料\博士材料\AT_tr_shortcut_pro\python_Data\work\I_AT3_data\tr_I_AT3_data.csv"   
tr_I_AT3_data = genfromtxt(tr_I_AT3_path, delimiter=',')
tr_I_AT3_x,tr_I_AT3_y=np.shape(tr_I_AT3_data)
tr_I_AT3_C=range(0, tr_I_AT3_x, 3)
tr_I_c_AT3_data = tr_I_AT3_data[tr_I_AT3_C,:]
tr_I_C_AT3_data=tr_I_c_AT3_data[:,0:8192]

tr_I_AT3_F=range(2, tr_I_AT3_x, 3)
tr_I_f_AT3_data = tr_I_AT3_data[tr_I_AT3_F,:]
tr_I_F_AT3_data=tr_I_f_AT3_data[:,0:8192]


lj_I_AT3_path = r"E:\学习资料\博士材料\AT_tr_shortcut_pro\python_Data\work\I_AT3_data\lj_I_AT3_data.csv"   
lj_I_AT3_data = genfromtxt(lj_I_AT3_path, delimiter=',')
lj_I_AT3_x,lj_I_AT3_y=np.shape(lj_I_AT3_data)
lj_I_AT3_C=range(0, lj_I_AT3_x, 3)
lj_I_c_AT3_data = lj_I_AT3_data[lj_I_AT3_C,:]
lj_I_C_AT3_data=lj_I_c_AT3_data[:,0:8192]

lj_I_AT3_F=range(2, lj_I_AT3_x, 3)
lj_I_f_AT3_data = lj_I_AT3_data[lj_I_AT3_F,:]
lj_I_F_AT3_data=lj_I_f_AT3_data[:,0:8192]


arc_I_AT3_path = r"E:\学习资料\博士材料\AT_tr_shortcut_pro\python_Data\work\I_AT3_data\arc_I_AT3_data.csv"   
arc_I_AT3_data = genfromtxt(arc_I_AT3_path, delimiter=',')
arc_I_AT3_x,arc_I_AT3_y=np.shape(arc_I_AT3_data)
arc_I_AT3_C=range(0, arc_I_AT3_x, 3)
arc_I_c_AT3_data = arc_I_AT3_data[arc_I_AT3_C,:]
arc_I_C_AT3_data=arc_I_c_AT3_data[:,0:8192]

arc_I_AT3_F=range(2, arc_I_AT3_x, 3)
arc_I_f_AT3_data = arc_I_AT3_data[arc_I_AT3_F,:]
arc_I_F_AT3_data=arc_I_f_AT3_data[:,0:8192]


'''
AT3电压数据导入  导入的数据为接触线、钢轨、正馈线
'''

tr_U_AT3_path = r"E:\学习资料\博士材料\AT_tr_shortcut_pro\python_Data\work\U_AT3_data\tr_U_AT3_data.csv"   
tr_U_AT3_data = genfromtxt(tr_U_AT3_path, delimiter=',')
tr_U_AT3_x,tr_U_AT3_y=np.shape(tr_U_AT3_data)
tr_U_AT3_C=range(0, tr_U_AT3_x, 3)
tr_U_c_AT3_data = tr_U_AT3_data[tr_U_AT3_C,:]
tr_U_C_AT3_data=tr_U_c_AT3_data[:,0:8192]

tr_U_AT3_F=range(2, tr_U_AT3_x, 3)
tr_U_f_AT3_data = tr_U_AT3_data[tr_U_AT3_F,:]
tr_U_F_AT3_data=tr_U_f_AT3_data[:,0:8192]


lj_U_AT3_path = r"E:\学习资料\博士材料\AT_tr_shortcut_pro\python_Data\work\U_AT3_data\lj_U_AT3_data.csv"   
lj_U_AT3_data = genfromtxt(lj_U_AT3_path, delimiter=',')
lj_U_AT3_x,lj_U_AT3_y=np.shape(lj_U_AT3_data)
lj_U_AT3_C=range(0, lj_U_AT3_x, 3)
lj_U_c_AT3_data = lj_U_AT3_data[lj_U_AT3_C,:]
lj_U_C_AT3_data=lj_U_c_AT3_data[:,0:8192]

lj_U_AT3_F=range(2, lj_U_AT3_x, 3)
lj_U_f_AT3_data = lj_U_AT3_data[lj_U_AT3_F,:]
lj_U_F_AT3_data=lj_U_f_AT3_data[:,0:8192]


arc_U_AT3_path = r"E:\学习资料\博士材料\AT_tr_shortcut_pro\python_Data\work\U_AT3_data\arc_U_AT3_data.csv"   
arc_U_AT3_data = genfromtxt(arc_U_AT3_path, delimiter=',')
arc_U_AT3_x,arc_U_AT3_y=np.shape(arc_U_AT3_data)
arc_U_AT3_C=range(0, arc_U_AT3_x, 3)
arc_U_c_AT3_data = arc_U_AT3_data[arc_U_AT3_C,:]
arc_U_C_AT3_data=arc_U_c_AT3_data[:,0:8192]

arc_U_AT3_F=range(2, arc_U_AT3_x, 3)
arc_U_f_AT3_data = arc_U_AT3_data[arc_U_AT3_F,:]
arc_U_F_AT3_data=arc_U_f_AT3_data[:,0:8192]



'''
小波包分解三维能量分布图，时频三维图
'''
def Reconstruction(data,n,N):#小波包系数重构，用于画三维能量分布图
    feature = []
    data=np.nan_to_num(data) #数据中存在nan，将nan转换为0，求范数时不影响最终^M
    wp = pywt.WaveletPacket(data=data, wavelet='db1',mode='symmetric',maxlevel=n) 
    add=wp['aaa'].data
    wpp = pywt.WaveletPacket(data=add, wavelet='db1',mode='symmetric',maxlevel=N)
    new_wpp= pywt.WaveletPacket(data=None, wavelet='db1',mode='symmetric',maxlevel=N)
    for i in [node.path for node in wpp.get_level(N, 'freq')]:
        new_wpp[i]=wpp[i].data
        feature.append(new_wpp.reconstruct(update=True))
        new_wpp= pywt.WaveletPacket(data=None, wavelet='db1',mode='symmetric',maxlevel=N)
    return feature
        
def my_perspective(data,n,nn): #data表示用于分解的波形数据，n表示小波包分解层数  
    feature = []
    data=np.nan_to_num(data) #数据中存在nan，将nan转换为0，求范数时不影响最终^M
    wp = pywt.WaveletPacket(data=data, wavelet='db1',mode='symmetric',maxlevel=n) 
    add=wp['aaa'].data
    wpp = pywt.WaveletPacket(data=add, wavelet='db1',mode='symmetric',maxlevel=nn)
    new_wpp= pywt.WaveletPacket(data=None, wavelet='db1',mode='symmetric',maxlevel=nn)
    for i in [node.path for node in wpp.get_level(nn, 'freq')]:
        new_wpp[i]=abs(wpp[i].data)
        feature.append(new_wpp.reconstruct(update=True))
        new_wpp= pywt.WaveletPacket(data=None, wavelet='db1',mode='symmetric',maxlevel=nn)
    N=np.power(2,nn)
    xshape=np.shape(add)[0]
    x=range(0, xshape, 1)
    y=range(0, N, 1) 
    z=np.array(feature)
    z=abs(z)

    x,y=np.meshgrid(x,y)
    fig = plt.figure(figsize=(8,6))
    ax=fig.gca(projection='3d')
    surf=ax.plot_surface(x,y,z,cmap=plt.cm.jet)
    ax.set_xlabel("x")
    ax.set_xlabel("y")
    ax.set_xlabel("z")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    return z

'''
小波包分解二维能量分布图，时频二维图
'''
def td_p(data,n,nn):
    z=my_perspective(data,n,nn)
    z=np.array(z)
    plt.pcolormesh(z,cmap=plt.cm.jet)
    plt.colorbar()
    plt.show()
    

    
    
    
    

'''
特征数据集建立
'''

def feat_creat(data,n,N):#n表示一次分解层数，N表示第二次分解层数
    x,y=np.shape(data)
    feature = []
    data=np.nan_to_num(data) #数据中存在nan，将nan转换为0，求范数时不影响最终
    for i in range(x):
        wp = pywt.WaveletPacket(data=data[i], wavelet='db1',mode='symmetric',maxlevel=n) 
        add=wp['aaa'].data
        feature.append(wave_ener_entropy(add,N))  
    return feature



'''
feature_lj_U_tran_C=np.array(feat_creat(lj_C_tran_data,3,7))
feature_tr_U_tran_C=np.array(feat_creat(tr_C_tran_data,3,7))
feature_arc_U_tran_C=np.array(feat_creat(arc_C_tran_data,3,7))

feature_lj_U_tran_F=np.array(feat_creat(lj_F_tran_data,3,7))
feature_tr_U_tran_F=np.array(feat_creat(tr_F_tran_data,3,7))
feature_arc_U_tran_F=np.array(feat_creat(arc_F_tran_data,3,7))


feature_lj_I_AT1_C=np.array(feat_creat(lj_I_C_AT1_data,3,7))
feature_tr_I_AT1_C=np.array(feat_creat(tr_I_C_AT1_data,3,7))
feature_arc_I_AT1_C=np.array(feat_creat(arc_I_C_AT1_data,3,7))

feature_lj_I_AT1_F=np.array(feat_creat(lj_I_F_AT1_data,3,7))
feature_tr_I_AT1_F=np.array(feat_creat(tr_I_F_AT1_data,3,7))
feature_arc_I_AT1_F=np.array(feat_creat(arc_I_F_AT1_data,3,7))


feature_lj_I_AT3_C=np.array(feat_creat(lj_I_C_AT3_data,3,7))
feature_tr_I_AT3_C=np.array(feat_creat(tr_I_C_AT3_data,3,7))
feature_arc_I_AT3_C=np.array(feat_creat(arc_I_C_AT3_data,3,7))

feature_lj_I_AT3_F=np.array(feat_creat(lj_I_F_AT3_data,3,7))
feature_tr_I_AT3_F=np.array(feat_creat(tr_I_F_AT3_data,3,7))
feature_arc_I_AT3_F=np.array(feat_creat(arc_I_F_AT3_data,3,7))

feature_lj_U_AT3_C=np.array(feat_creat(lj_U_C_AT3_data,3,7))
feature_tr_U_AT3_C=np.array(feat_creat(tr_U_C_AT3_data,3,7))
feature_arc_U_AT3_C=np.array(feat_creat(arc_U_C_AT3_data,3,7))

feature_lj_U_AT3_F=np.array(feat_creat(lj_U_F_AT3_data,3,7))
feature_tr_U_AT3_F=np.array(feat_creat(tr_U_F_AT3_data,3,7))
feature_arc_U_AT3_F=np.array(feat_creat(arc_U_F_AT3_data,3,7))
'''


'''
时频图特征数据集建立
'''
def feat_perspective(data,n,N):#n表示一次分解层数，N表示第二次分解层数
    x,y=np.shape(data)
    feature = []
    feature_zhh=[]
    data=np.nan_to_num(data) #数据中存在nan，将nan转换为0，求范数时不影响最终
    for i in range(x):
        wp = pywt.WaveletPacket(data=data[i], wavelet='db1',mode='symmetric',maxlevel=n) 
        add=wp['aaa'].data
        
        wpp = pywt.WaveletPacket(data=add, wavelet='db1',mode='symmetric',maxlevel=N)
        new_wpp= pywt.WaveletPacket(data=None, wavelet='db1',mode='symmetric',maxlevel=N)
        for i in [node.path for node in wpp.get_level(N, 'freq')]:
            new_wpp[i]=wpp[i].data
            feature.extend(new_wpp.reconstruct(update=True))
            new_wpp= pywt.WaveletPacket(data=None, wavelet='db1',mode='symmetric',maxlevel=N)
            feature_Norm=feature/np.sum(feature)
        feature_zhh.append(feature_Norm)
        feature=[]
    return feature_zhh
'''
a=feat_perspective(lj_train_data,3,7)
a=np.array(a)

b=feat_perspective(arc_train_data,3,7)
b=np.array(b)

c=feat_perspective(tr_train_data,3,7)
c=np.array(c)
'''
'''
机器学习分类建模  000表示雷击，010表示短路，100表示电弧
'''
'''
feature_lj=np.hstack((feature_lj_U_tran_C,feature_lj_U_tran_F,feature_lj_I_AT1_C,feature_lj_I_AT1_F,feature_lj_I_AT3_C,feature_lj_I_AT3_F,feature_lj_U_AT3_C,feature_lj_U_AT3_F))
feature_tr=np.hstack((feature_tr_U_tran_C,feature_tr_U_tran_F,feature_tr_I_AT1_C,feature_tr_I_AT1_F,feature_tr_I_AT3_C,feature_tr_I_AT3_F,feature_tr_U_AT3_C,feature_tr_U_AT3_F))
feature_arc=np.hstack((feature_lj_U_tran_C,feature_arc_U_tran_F,feature_arc_I_AT1_C,feature_arc_I_AT1_F,feature_arc_I_AT3_C,feature_arc_I_AT3_F,feature_arc_U_AT3_C,feature_arc_U_AT3_F))

feature=np.vstack((feature_lj,feature_tr,feature_arc))#矩阵合并
y=(np.vstack((lj_c_tran_data,tr_c_tran_data,arc_c_tran_data)))[:,8192:8195]#分类标签


feature_array = np.arange(feature.shape[0])
np.random.shuffle(feature_array)

test_feature=feature[feature_array[0:100]]
train_feature=feature[feature_array[100:738]]


test_y=y[feature_array[0:100]]
train_y=y[feature_array[100:738]]
'''
'''
多项式分布  0表示雷击，1表示短路，2表示电弧
'''
'''
import pandas as pd

gg = pd.DataFrame(train_feature)

writer = pd.ExcelWriter('train_feature.xlsx')		# 写入Excel文件
gg.to_excel(writer, 'page_1', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
writer.save()

writer.close()
'''
def wgn(x, snr):
    Ps = np.sum(abs(x)**2)/len(x)
    Pn = Ps/(10**((snr/10)))
    noise = np.random.randn(len(x)) * np.sqrt(Pn)
    signal_add_noise = x + noise
    return signal_add_noise

def noisfeat_creat(data,n,N):#n表示一次分解层数，N表示第二次分解层数
    x,y=np.shape(data)
    feature = []
    data=np.nan_to_num(data) #数据中存在nan，将nan转换为0，求范数时不影响最终
    for i in range(x):
        wp = pywt.WaveletPacket(data=wgn(data[i],26.5), wavelet='db1',mode='symmetric',maxlevel=n) 
        add=wp['aaa'].data
        feature.append(wave_ener_entropy(add,N))  
    return feature

def feat_creat_spect(data,n,N):#n表示一次分解层数，N表示第二次分解层数
    x,y=np.shape(data)
    feature = []
    data=np.nan_to_num(data) #数据中存在nan，将nan转换为0，求范数时不影响最终
    for i in range(x):
        wp = pywt.WaveletPacket(data=data[i], wavelet='db1',mode='symmetric',maxlevel=n) 
        add=wp['aaa'].data
        feature.append(wave_ener_spect(add,N))  
    return feature    


nos_arc_I_C_AT1_data=wgn(arc_I_C_AT1_data[0], 26.5)


noisfeature_lj_U_tran_C=np.array(noisfeat_creat(lj_C_tran_data,3,7))
noisfeature_tr_U_tran_C=np.array(noisfeat_creat(tr_C_tran_data,3,7))
noisfeature_arc_U_tran_C=np.array(noisfeat_creat(arc_C_tran_data,3,7))

noisfeature_lj_U_tran_F=np.array(noisfeat_creat(lj_F_tran_data,3,7))
noisfeature_tr_U_tran_F=np.array(noisfeat_creat(tr_F_tran_data,3,7))
noisfeature_arc_U_tran_F=np.array(noisfeat_creat(arc_F_tran_data,3,7))


noisfeature_lj_I_AT1_C=np.array(noisfeat_creat(lj_I_C_AT1_data,3,7))
noisfeature_tr_I_AT1_C=np.array(noisfeat_creat(tr_I_C_AT1_data,3,7))
noisfeature_arc_I_AT1_C=np.array(noisfeat_creat(arc_I_C_AT1_data,3,7))

noisfeature_lj_I_AT1_F=np.array(noisfeat_creat(lj_I_F_AT1_data,3,7))
noisfeature_tr_I_AT1_F=np.array(noisfeat_creat(tr_I_F_AT1_data,3,7))
noisfeature_arc_I_AT1_F=np.array(noisfeat_creat(arc_I_F_AT1_data,3,7))


noisfeature_lj_I_AT3_C=np.array(noisfeat_creat(lj_I_C_AT3_data,3,7))
noisfeature_tr_I_AT3_C=np.array(noisfeat_creat(tr_I_C_AT3_data,3,7))
noisfeature_arc_I_AT3_C=np.array(noisfeat_creat(arc_I_C_AT3_data,3,7))

noisfeature_lj_I_AT3_F=np.array(noisfeat_creat(lj_I_F_AT3_data,3,7))
noisfeature_tr_I_AT3_F=np.array(noisfeat_creat(tr_I_F_AT3_data,3,7))
noisfeature_arc_I_AT3_F=np.array(noisfeat_creat(arc_I_F_AT3_data,3,7))

noisfeature_lj_U_AT3_C=np.array(noisfeat_creat(lj_U_C_AT3_data,3,7))
noisfeature_tr_U_AT3_C=np.array(noisfeat_creat(tr_U_C_AT3_data,3,7))
noisfeature_arc_U_AT3_C=np.array(noisfeat_creat(arc_U_C_AT3_data,3,7))

noisfeature_lj_U_AT3_F=np.array(noisfeat_creat(lj_U_F_AT3_data,3,7))
noisfeature_tr_U_AT3_F=np.array(noisfeat_creat(tr_U_F_AT3_data,3,7))
noisfeature_arc_U_AT3_F=np.array(noisfeat_creat(arc_U_F_AT3_data,3,7))


noisfeature_lj=np.hstack((noisfeature_lj_U_tran_C,noisfeature_lj_U_tran_F,noisfeature_lj_I_AT1_C,noisfeature_lj_I_AT1_F,noisfeature_lj_I_AT3_C,noisfeature_lj_I_AT3_F,noisfeature_lj_U_AT3_C,noisfeature_lj_U_AT3_F))
noisfeature_tr=np.hstack((noisfeature_tr_U_tran_C,noisfeature_tr_U_tran_F,noisfeature_tr_I_AT1_C,noisfeature_tr_I_AT1_F,noisfeature_tr_I_AT3_C,noisfeature_tr_I_AT3_F,noisfeature_tr_U_AT3_C,noisfeature_tr_U_AT3_F))
noisfeature_arc=np.hstack((noisfeature_lj_U_tran_C,noisfeature_arc_U_tran_F,noisfeature_arc_I_AT1_C,noisfeature_arc_I_AT1_F,noisfeature_arc_I_AT3_C,noisfeature_arc_I_AT3_F,noisfeature_arc_U_AT3_C,noisfeature_arc_U_AT3_F))

noisfeature=np.vstack((noisfeature_lj,noisfeature_tr,noisfeature_arc))#矩阵合并
noisy=(np.vstack((lj_c_tran_data,tr_c_tran_data,arc_c_tran_data)))[:,8192:8195]#分类标签


noisfeature_array = np.arange(noisfeature.shape[0])
np.random.shuffle(noisfeature_array)

noistest_feature=noisfeature[noisfeature_array[0:100]]
noistrain_feature=noisfeature[noisfeature_array[100:738]]


noistest_y=noisy[noisfeature_array[0:100]]
noistrain_y=noisy[noisfeature_array[100:738]]


plt.plot(noistest_feature[2],color="green")
plt.show()


