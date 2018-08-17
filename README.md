# Asset-Allocation-
# -*- coding: UTF-8 -*-
import pandas as pd
import bt
import ds
import seaborn as sns
from matplotlib import pyplot as plt
from cvxopt import solvers, matrix
import numpy as np
import cvxopt
from scipy.optimize import minimize
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['font.serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
%matplotlib inline

Ori=pd.read_excel('data_source4.xlsx')
Ori = Ori.dropna()

df=Ori.set_index('date')
df.head()
df2 = df.pct_change().fillna(0)


df3 = (df2+1).cumprod()


df3.plot(figsize = (12,6))

ax=df3.plot(title='5年期各指数走势图',figsize=(12,5),rot=0)
type(ax)
vals=ax.get_yticks()
ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])
plt.savefig('走势',dpi=199)

r=df3.iloc[:,[0,1,2,3,6,8,9]]
r.plot(figsize=(12,5))
r2=df3.iloc[:,[4,5,7]]
r2.plot(figsize=(12,5))

# df3[:2]=format(df3[:2],'.0%')

f,ax = plt.subplots(figsize=(12,5))
cor=df.corr()
cmap = sns.cubehelix_palette(start = 1, rot = 3, gamma=0.8, as_cmap = True)
sns.heatmap(cor, annot=True,annot_kws={'size':15,'weight':'bold'},cmap ='Purples',linewidths = 0.05, ax = ax)
ax.set_title('指数收益率相关系数')
plt.xticks(fontsize = 15,rotation = 30)
plt.yticks(fontsize = 15,rotation = 30)

frm_dts = list(df.index)

dts1 = []
for i in range(0, len(frm_dts)):
    if (i % 90 == 0 and i != 0):
        #frm_dts[i]=frm_dts[i].to_pydatetime().strftime('%Y-%m-%d')
        dts1.append(frm_dts[i])
dts1

dts2 = []
for i in range(0, len(frm_dts)):
    if (i % 30 == 0 and i != 0):
        #frm_dts[i]=frm_dts[i].to_pydatetime().strftime('%Y-%m-%d')
        dts2.append(frm_dts[i])


def cal_pnl(retFrm,weightsDict):#收益率dataframe,资产间权重,根据每一期的资产权重和资产收益来计算投资组合净值
    retFrmList = []
    dts = list(weightsDict.keys())
    for i in range(len(dts)):
        weights = weightsDict[dts[i]]
        if i < len(dts) - 1:
            subFrm = retFrm.loc[dts[i]:dts[i+1],:]
        else:
            subFrm = retFrm.loc[dts[i]:,:]
        ret = np.sum((1+subFrm).cumprod()*weights,axis=1)
        if i != 0:
            ret = ret[1:]*retFrmList[i-1].values[-1]
        retFrmList.append(ret)
    nav = pd.concat(retFrmList)
    nav.index = pd.to_datetime(nav.index)
    return nav

df2.head()

# a1 = []
# retFrmList2 = []
# for i in range(0,12):
#     for j in range(0,9):
#         weights = capm1_dict[dts1[i]]
#         subFrm = df2.loc[dts1[i]:dts1[i+1],:].pct_change()
#         subFrm.replace([np.inf,-np.inf],np.nan)
#         ret =subFrm.fillna(0).iloc[:,j:j+1]*weights[j]
#         ret2=np.sum(subFrm.fillna(0)*weights,axis=1)
#         a1.append(ret)
#         retFrmList2.append(ret2)

# combined_df = a1[0]
# for i in range(9, len(a1), 9):
#     print(i)
#     combined_df = pd.concat([combined_df, a1[i]])
# combined_df

# combined_retFrm = retFrmList2[0]
# for i in range(9, len(a1), 9):
#     print(i)
#     combined_retFrm = pd.concat([combined_retFrm, retFrmList2[i]])
# combined_retFrm

# x=(combined_df.iloc[:,0]/combined_retFrm).fillna(0)

# a2=np.array(a2)
# a3=np.array(a3)

# import pickle
# pickle.dump(a1, open('a1.pkl','wb'))
# pickle.dump(retFrmList2, open('retFrmList2.pkl','wb'))

# a2.iloc[:,0]/a3

# for i in range(19,108,9):
#         a2=a1[9].append(a1[i-1])
#         a3=retFrmList2[9].append(retFrmList2[i-1])

# (a2.iloc[:,0]/a3).dropna().plot()

# m2=retFrmList2[0].append(retFrmList2[1:])

# m2.replace([np.inf,-np.inf],np.nan).dropna().plot()

# capm1_ret_0 = cal_pn2(df2,capm1_dict)



# m=o1[0]
# m2=m.append(o1[1:])

# k4=[]
# for j in range (0,9),i in range(0,12):
#         subFrm = df2.loc[dts1[i]:dts1[i+1],:]
#         ret = np.sum((1+subFrm).cumprod()*capm1_dict[dts1[i]],axis=1)
#         ret_1 = np.sum((1+subFrm).iloc[:,j:j+1].cumprod()*capm1_dict[dts1[i]][j],axis=1)
#         k3=ret_1/ret
#         k4.append(k3)
# k4

# subFrm = df2.loc[dts1[0]:dts1[1],:]
# ret_1 = np.sum((1+subFrm).iloc[:,0:0+1].cumprod()*capm1_dict[dts1[0]][0],axis=1)
# ret_1

def capm_objective(x_weight,subret,risk_coef = 1):
    mean = np.matrix(subret.mean()).T
    one_cov_matrix = np.matrix(subret.cov())
    x_weight = np.matrix(x_weight).T#转化为默认列向量
    #return -(x_weight.T*mean/(risk_coef * x_weight.T*one_cov_matrix*x_weight/2))#添加负数符号最大化转为最小化
    #return risk_coef * x_weight.T*one_cov_matrix*x_weight/2
    return -(x_weight.T*mean - (risk_coef * x_weight.T*one_cov_matrix*x_weight/2))
    #return -x_weight.T*me an #返回值的数量级会影响函数的迭代次数，设置'ftol'
    
# 优化问题的第一个约束条件  
def constraint1(x_weight):  
    return np.sum(x_weight) - 1.0  
  
# 优化问题的第二个约束条件  
def constraint2(x_weight):  
    return x_weight - 1e-10

def constraint3(x_weight):
    return - ((x_weight * np.array([0,1,0,0,0,0,0,0,1,0])).sum() - 0.5)

def constraint4(x_weight):
    return -((x_weight * np.array([0,0,0,0,1,1,0,1,0,0])).sum() - 0.6)

def constraint5(x_weight):
    return -((x_weight * np.array([0,0,0,0,0,0,0,0,0,1])).sum() - 0.4)

def constraint6(x_weight):
    return -((x_weight * np.array([0,0,0,0,0,0,1,0,0,0])).sum() - 0.3)

cons=({'type': 'eq', 'fun': constraint1},{'type': 'ineq', 'fun': constraint2},{'type': 'ineq', 'fun': constraint3}, {'type': 'ineq', 'fun': constraint4},{'type': 'ineq', 'fun': constraint5},{'type': 'ineq', 'fun': constraint6})
#       {'type': 'ineq', 'fun': constraint4},{'type': 'ineq', 'fun': constraint5})

### rbf1,2,2

capm11_dict = {}
frm_dts = list(df.index)
weight0=[1/10]*10
for dt in dts2:#dts为每季末
    idx = frm_dts.index(dt)
    #bnds = ((0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None))
    subFrm = df.pct_change().fillna(0)[idx-25:idx]
    res= minimize(capm_objective, weight0, args=(subFrm,40), method='SLSQP',constraints=cons, options={'disp': True, 'maxiter': 100, 'ftol': 1e-10})  
    capm11_dict[dt] = res.x
    #bounds = bnds

capm11_ret = cal_pnl(df2,capm11_dict)

capm11_ret.plot(title='rbf(1,2,2)',figsize=(12,5))

### rbf (2,2,1)

capm12_dict = {}
frm_dts = list(df.index)
weight0=[1/10]*10
for dt in dts2:#dts为每季末
    idx = frm_dts.index(dt)
    #bnds = ((0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None))
    subFrm = df.pct_change().fillna(0)[idx-25:idx]
    res= minimize(capm_objective, weight0, args=(subFrm,66), method='SLSQP',constraints=cons, options={'disp': True, 'maxiter': 100, 'ftol': 1e-10})  
    capm12_dict[dt] = res.x
    #bounds = bnds

capm12_ret = cal_pnl(df2,capm12_dict)

capm12_ret.plot(title='rbf',figsize=(12,5))

### Mean-Variance Risk=10

capm1_dict = {}
frm_dts = list(df.index)
weight0=[1/10]*10
for dt in dts1:#dts为每季末
    idx = frm_dts.index(dt)
    #bnds = ((0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None))
    subFrm = df.pct_change().fillna(0)[idx-70:idx]
    res= minimize(capm_objective, weight0, args=(subFrm,10), method='SLSQP',constraints=cons, options={'disp': True, 'maxiter': 100, 'ftol': 1e-10})  
    capm1_dict[dt] = res.x
    #bounds = bnds

capm1_ret = cal_pnl(df2,capm1_dict)
# capm1_ret.plot()

# capm1_ret_0 = cal_pn2(df2,capm1_dict)
# capm1_ret_0

capm1_ret.plot(figsize=(12,5))

frmList = []
for i in capm1_dict.keys():
    frmList.append(pd.DataFrame(capm1_dict[i],index = df2.columns,columns = [i]))
weightfrm = pd.concat(frmList,axis=1).T

weightfrm.plot.area(title='Mean-Variance risk preference =10',figsize = (12,5),colormap = 'Paired_r')

### Mean-Variance risk=5

capm2_dict = {}
frm_dts = list(df.index)
weight0=[1/10]*10
for dt in dts1:#dts为每季末
    idx = frm_dts.index(dt)
    #bnds = ((0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None))
    subFrm = df.pct_change().fillna(0)[idx-70:idx]
    res= minimize(capm_objective, weight0, args=(subFrm,5), method='SLSQP',constraints=cons, options={'disp': True, 'maxiter': 100, 'ftol': 1e-10})  
    capm2_dict[dt] = res.x
    #bounds = bnds

frmList = []
for i in capm2_dict.keys():
    frmList.append(pd.DataFrame(capm2_dict[i],index = df2.columns,columns = [i]))
weightfrm = pd.concat(frmList,axis=1).T

weightfrm.plot.area(title='Mean-Variance risk preference =5',figsize = (12,5),colormap = 'Paired_r')

capm2_ret = cal_pnl(df2,capm2_dict)

capm2_ret.plot(title=' Mean-Variance risk preference =5',figsize=(12,5))

(pow(capm2_ret.ix[len(capm2_ret.index) - 1] / capm2_ret.ix[0], 250/1000)-1)*100

### Mean-Variance Risk=1

capm3_dict = {}
frm_dts = list(df.index)
weight0=[1/10]*10
for dt in dts1:#dts为每季末
    idx = frm_dts.index(dt)
    #bnds = ((0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None))
    subFrm = df.pct_change().fillna(0)[idx-70:idx]
    res= minimize(capm_objective, weight0, args=(subFrm,1), method='SLSQP',constraints=cons, options={'disp': True, 'maxiter': 100, 'ftol': 1e-10})  
    capm3_dict[dt] = res.x
    #bounds = bnds

# risk_parity_list = []
# res3=[]
# frm_dts = list(df.index)
# weight0=[1/10]*10
# for dt in dts1:#dts为每季末
#     idx = frm_dts.index(dt)
#     subFrm = df.pct_change().fillna(0)[idx-70:idx]
#     res= minimize(capm_objective, weight0, args=(subFrm,1), method='SLSQP',constraints=cons, options={'disp': True, 'maxiter': 100, 'ftol': 1e-10})
#     risk_parity_list.append(res.x)
#     one_cov_matrix = np.matrix(subFrm.cov())
#     res2 = res.x*np.array(one_cov_matrix*np.matrix(res.x).T).T[0]
#     res3.append(res2)

# juzhen = np.array(risk_parity_list)
# res4=np.array(res3)

# rc_dataframe=pd.DataFrame(res4[0:14],index=dts1,columns=df.columns)
# rc_dataframe.plot(kind='bar',figsize=(12,5))

frmList = []
for i in capm3_dict.keys():
    frmList.append(pd.DataFrame(capm3_dict[i],index = df2.columns,columns = [i]))
weightfrm = pd.concat(frmList,axis=1).T

weightfrm.plot.area(title='Mean-Variance risk preference =1',figsize = (12,5),colormap = 'Paired_r')

weightfrm.plot.area(title='rbf',figsize = (12,5),colormap = 'Paired_r')

weightfrm.plot.area(title='MANN',figsize = (12,6),colormap = 'Paired_r')

capm3_ret = cal_pnl(df2,capm3_dict)

capm3_ret.plot(title=' Mean-Variance risk preference =1',figsize=(12,5))

(pow(capm3_ret.ix[len(capm3_ret.index) - 1] / capm3_ret.ix[0], 250/1000)-1)*100

### Mean-Variance risk=100

capm4_dict = {}
frm_dts = list(df.index)
weight0=[1/10]*10
for dt in dts1:#dts为每季末
    idx = frm_dts.index(dt)
    #bnds = ((0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None))
    subFrm = df.pct_change().fillna(0)[idx-70:idx]
    res= minimize(capm_objective, weight0, args=(subFrm,100), method='SLSQP',constraints=cons, options={'disp': True, 'maxiter': 100, 'ftol': 1e-10})  
    capm4_dict[dt] = res.x
    #bounds = bnds

frmList = []
for i in capm4_dict.keys():
    frmList.append(pd.DataFrame(capm4_dict[i],index = df2.columns,columns = [i]))
weightfrm = pd.concat(frmList,axis=1).T

weightfrm.plot.area(title='Mean-Variance risk preference =100',figsize = (12,5),colormap = 'Paired_r')

capm4_ret = cal_pnl(df2,capm4_dict)

capm4_ret.plot(title='Mean-Variance risk preference =100',figsize=(12,5))

(pow(capm4_ret.ix[len(capm4_ret.index) - 1] / capm4_ret.ix[0], 250/1000)-1)*100

# index_j = np.argmax(np.maximum.accumulate(capm4_ret) - capm4_ret)  # 结束位置
# print(index_j)
# index_i = np.argmax(capm4_ret[:index_j])  # 开始位置
# print(index_i)
# maxmium_drawdown = capm4_ret[index_j] - capm4_ret[index_i]  # 最大回撤
# print(maxmium_drawdown)


capm_ret = pd.concat([capm1_ret,capm2_ret,capm3_ret,capm4_ret],axis=1)
capm_ret.columns = ['风险系数10','风险系数5','风险系数1','风险系数100']
capm_ret = capm_ret.dropna()
#最小方差等价于风险系数无限大

capm_ret.plot(title='净值走势',figsize = (10,5))

### most diversified

def func_objective(x_weight,subret):
    std = subret.std().values
    one_cov_matrix = np.matrix(subret.cov())
    x_weight = np.matrix(x_weight)
    res = np.array(one_cov_matrix*x_weight.T).T[0]/std
    mean = res.mean()
    return np.square(res - mean).sum()

    
# 优化问题的第一个约束条件  
def constraint1(x_weight):  
    return np.sum(x_weight) - 1.0  
  
# 优化问题的第二个约束条件  
def constraint2(x_weight):  
    return x_weight - 1e-10

def constraint3(x_weight):
    return - ((x_weight * np.array([0,0,0,0,0,0,0,0,1,0])).sum() - 0.2)

def constraint4(x_weight):
     return -((x_weight * np.array([0,0,0,0,1,1,0,1,0,0])).sum() - 0.5)

def constraint5(x_weight):
     return -((x_weight * np.array([0,0,0,0,0,0,1,0,0,0])).sum() - 0.2)

def constraint6(x_weight):
     return -((x_weight * np.array([0,0,0,0,0,0,0,0,0,1])).sum() - 0.3)

cons=({'type': 'eq', 'fun': constraint1},{'type': 'ineq', 'fun': constraint2}, {'type': 'ineq', 'fun': constraint3}, {'type': 'ineq', 'fun': constraint4},{'type': 'ineq', 'fun': constraint5},{'type': 'ineq', 'fun': constraint6})
# #       {'type': 'ineq', 'fun': constraint4},{'type': 'ineq', 'fun': constraint5})

# dts2 = []
# for i in range(0, len(frm_dts)):
#     if (i % 30 == 0 and i != 0):
#         #frm_dts[i]=frm_dts[i].to_pydatetime().strftime('%Y-%m-%d')
#         dts2.append(frm_dts[i])


diversified_dict = {}
frm_dts = list(df.index)
weight0=[1/10]*10
for dt in dts1:#dts为每季末
    idx = frm_dts.index(dt)
    #bnds = ((0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None))
    subFrm = df.pct_change().fillna(0)[idx-89:idx]
    res= minimize(func_objective, weight0, args=subFrm, method='SLSQP',constraints=cons, options={'disp': True, 'maxiter': 150, 'ftol': 1e-16})  
    diversified_dict[dt] = res.x

diversified_ret = cal_pnl(df2,diversified_dict)

diversified_ret.plot(title='最大分散化模型',figsize=(12,5))

frmList = []
for i in diversified_dict.keys():
    frmList.append(pd.DataFrame(diversified_dict[i],index = df2.columns,columns = [i]))
weightfrm = pd.concat(frmList,axis=1).T

weightfrm.plot.area(title='最大分散化模型',figsize = (12,5),colormap = 'Paired_r')

annual_return=(pow(diversified_ret.ix[len(diversified_ret.index) - 1] / diversified_ret.ix[0], 250/1000)-1)*100
annual_return

# annual_return2=(pow(risk_parity_ret.ix[len(risk_parity_ret.index) - 1] / risk_parity_ret.ix[0], 250/1000)-1)
# annual_return2

# mf.columns=['ret']
# mf.index


# #print(df.loc[df['B'].isin(['one','three'])])
# mf.loc[mf.index.isin(['2018-01-02','2018-07-26'])]

# pow(risk_parity_ret['2018-07-26']/risk_parity_ret['2018-01-02'],250/len()

# index_j = np.argmax(np.maximum.accumulate(diversified_ret) - diversified_ret)  # 结束位置
# print(index_j)
# index_i = np.argmax(diversified_ret[:index_j])  # 开始位置
# print(index_i)
# maxmium_drawdown = diversified_ret[index_j] - diversified_ret[index_i]  # 最大回撤
# print(maxmium_drawdown)



# rf=0.0284 #10年期国债年化收益率#
# sharp_ratio=(annual_return2-rf)/vol
# sharp_ratio

# annual_mean=risk_parity_ret2.mean()*250
# annual_mean

# info_ratio=annual_mean/vol
# info_ratio

# type(risk_parity_ret)

# ben=pd.read_excel('benchmark5.xlsx')
# ben=ben[['benchmark(2/8)','benchmark(4/6)']]
# ben2=ben.iloc[90:,:]
# ben2=ben2.pct_change().fillna(0)
# ben3=(ben2+1).cumprod()



# benchmark_rtn=ben.pct_change().fillna(0)
# benchmark_rtn=benchmark_rtn.iloc[90:,:]
# b1=ben.iloc[:,0]
# b2=ben.iloc[:,1]

# #Benchmark
# #Benchmark1 20%hs300 80%zz50 
# beta1=risk_parity_ret2.cov(benchmark_rtn.iloc[:,0])/benchmark_rtn.iloc[:,0].var()
# beta1


# beta2=risk_parity_ret2.cov(benchmark_rtn.iloc[:,1])/benchmark_rtn.iloc[:,1].var()
# beta2

# beta = df['rtn'].cov(df['benchmark_rtn']) / df['benchmark_rtn'].var()

# annual_index1=(pow(b1.ix[len(b1.index) - 1] / b1.ix[0], 250/1183)-1)
# annual_index2=(pow(b2.ix[len(b2.index) - 1] / b2.ix[0], 250/1183)-1)

# a1 = (annual_return2 - rf) - beta1 * (annual_index1 - rf)  # 计算alpha值
# a2=(annual_return2 - rf) - beta2 * (annual_index2 - rf) 
# print(a1,a2)

# d={'YEAR':[2017,2018],'年化收益率':[annual_return,annual_return2],'波动率':[vol,0.075]}

# mf=pd.DataFrame(data=d)

# mf.set_index('YEAR')

##多年期回测 



### risk parity

# 计算每类资产对总资产组合的风险贡献  
def risk_budget_objective(weight,subFrm):  
    subFrm = subFrm.apply(lambda s:(s-s.mean())/s.std())#将收益率标准化
    one_cov_matrix = np.matrix(subFrm.cov())
    res = weight*np.array(one_cov_matrix*np.matrix(weight).T).T[0]
    mean = res.mean()
    return np.square(res - mean).sum()#标准化之后的res不会太小

# 优化问题的第一个约束条件  
def constraint1(x_weight):  
    return np.sum(x_weight)-1.0  

# 优化问题的第二个约束条件  
def constraint2(x_weight):  
    return x_weight - 1e-16

def constraint3(res):
    return res - 1e-16

cons=({'type': 'eq', 'fun': constraint1},{'type': 'ineq', 'fun': constraint2},{'type': 'ineq', 'fun': constraint3})

risk_parity_dict = {}
frm_dts = list(df.index)
weight0=[1/10]*10
for dt in dts1:#dts为每季末
    idx = frm_dts.index(dt)
    subFrm = df.pct_change().fillna(0)[idx-70:idx]
    res= minimize(risk_budget_objective, weight0, args=subFrm, method='SLSQP',constraints=cons, options={'disp': True, 'maxiter': 100, 'ftol': 1e-17})  
    risk_parity_dict[dt] = res.x

# risk_parity_list = []
# frm_dts = list(df.index)
# weight0=[1/10]*10
# for dt in dts1:#dts为每季末
#     idx = frm_dts.index(dt)
#     subFrm = df.pct_change().fillna(0)[idx-70:idx]
#     res= minimize(risk_budget_objective, weight0, args=subFrm, method='SLSQP',constraints=cons, options={'disp': True, 'maxiter': 100, 'ftol': 1e-16})  
#     risk_parity_list.append(res.x)

# juzhen = np.array(risk_parity_list)
# print(juzhen)

# risk_parity_list = []
# res3=[]
# frm_dts = list(df.index)
# weight0=[1/10]*10
# for dt in dts1:#dts为每季末
#     idx = frm_dts.index(dt)
#     subFrm = df.pct_change().fillna(0)[idx-89:idx]
#     res= minimize(risk_budget_objective, weight0, args=subFrm, method='SLSQP',constraints=cons, options={'disp': True, 'maxiter': 200, 'ftol': 1e-16})  
#     risk_parity_list.append(res.x)
#     one_cov_matrix = np.matrix(subFrm.cov())
#     res2 = res.x*np.array(one_cov_matrix*np.matrix(res.x).T).T[0]
#     res3.append(res2)

# juzhen = np.array(risk_parity_list)
# res4=np.array(res3)


# rc_dataframe=pd.DataFrame(res4[0:14],index=dts1,columns=df.columns)
# rc_dataframe.plot(figsize=(12,5))

# len(dts1)

# weight_dataframe=pd.DataFrame(juzhen[0:14],index=dts1,columns=df.columns)


risk_parity_ret = cal_pnl(df2,risk_parity_dict)

risk_parity_ret.plot(title='Risk Parity风险平价', figsize=(12,5))

frmList = []
for i in diversified_dict.keys():
    frmList.append(pd.DataFrame(risk_parity_dict[i],index = df2.columns,columns = [i]))
weightfrm = pd.concat(frmList,axis=1).T

weightfrm.plot.area(title='Risk Parity风险平价',figsize = (12,5),colormap = 'Paired_r')

(pow(risk_parity_ret.ix[len(risk_parity_ret.index) - 1] / risk_parity_ret.ix[0], 250/1000)-1)*100



### 试用

dts3 = []
for i in range(0, len(frm_dts)):
    if (i % 50 == 0 and i != 0):
        #frm_dts[i]=frm_dts[i].to_pydatetime().strftime('%Y-%m-%d')
        dts3.append(frm_dts[i])

risk_parity_dict = {}
frm_dts = list(df.index)
weight0=[1/10]*10
for dt in dts3:#dts为每季末
    idx = frm_dts.index(dt)
    subFrm = df.pct_change().fillna(0)[idx-40:idx]
    res= minimize(risk_budget_objective, weight0, args=subFrm, method='SLSQP',constraints=cons, options={'disp': True, 'maxiter': 100, 'ftol': 1e-17})  
    risk_parity_dict[dt] = res.x

risk_parity_ret = cal_pnl(df2,risk_parity_dict)

risk_parity_ret.to_excel('risk_parity.xlsx')

risk_parity_ret.plot(title='Risk Parity风险平价', figsize=(12,5))

(pow(risk_parity_ret.ix[len(risk_parity_ret.index) - 1] / risk_parity_ret.ix[0], 250/1000)-1)*100







# 计算每类资产对总资产组合的风险贡献  
def pca_objective(weight,subFrm):  
    #subFrm = subFrm.apply(lambda s:(s-s.mean())/s.std())
    one_cov_matrix = np.matrix(subFrm.cov())
    eigval,eigvec=np.linalg.eig(one_cov_matrix)
    E = np.matrix(eigvec).T
    weight=np.matrix(weight)
    res = np.array(E.T*weight.T).T[0]*np.array(E.T*one_cov_matrix*weight.T).T[0]
    mean = res.mean()
    return np.square(res - mean).sum()#标准化之后的res不会太小


# 优化问题的第一个约束条件  
def constraint1(x_weight):  
    return np.sum(x_weight)-1.0  
  
# 优化问题的第二个约束条件  
def constraint2(x_weight):  
    return x_weight - 1e-10

# def constraint3(x_weight):
#     return - ((x_weight * np.array([1,1,1,0,0,1,0,0])).sum() - 0.75)

# def constraint4(x_weight):
#     return - ((x_weight * np.array([1,1,1,0,0,1,0,0])).sum() - 0.6)

# def constraint5(x_weight):
#     return -((x_weight * np.array([0,0,0,1,1,0,1,0])).sum() - 0.3)

# def constraint6(x_weight):
#     return ((x_weight * np.array([0,0,0,1,1,0,1,0])).sum() - 0.2)

# def constraint7(x_weight):
#     return -((x_weight * np.array([0,0,0,0,0,0,0,1])).sum() - 0.1)

# def constraint8(x_weight):
#     return -((x_weight * np.array([0,0,0,0,0,1,0,0])).sum() - 0.15)    

cons=({'type': 'eq', 'fun': constraint1},{'type': 'ineq', 'fun': constraint2})
#       {'type': 'ineq', 'fun': constraint3},
#       {'type': 'ineq', 'fun': constraint4},{'type': 'ineq', 'fun': constraint5},{'type': 'ineq', 'fun': constraint6},{'type': 'ineq', 'fun': constraint7},{'type': 'ineq', 'fun': constraint8})

pca_dict = {}
frm_dts = list(df.index)
weight0=[1/10]*10
for dt in dts1:#dts为每季末
    idx = frm_dts.index(dt)
    subFrm = df.pct_change().fillna(0)[idx-80:idx]
    #将最大跌代次数放宽至200次
    res= minimize(pca_objective, weight0, args=subFrm, method='SLSQP',constraints=cons, options={'disp': True, 'maxiter': 200, 'ftol': 1e-20})
    pca_dict[dt] = res.x
    


# pca_list = []
# res3=[]
# frm_dts = list(df.index)
# weight0=[1/10]*10
# for dt in dts1:#dts为每季末
#     idx = frm_dts.index(dt)
#     subFrm = df.pct_change().fillna(0)[idx-70:idx]
#     res= minimize(pca_objective, weight0, args=subFrm, method='SLSQP',constraints=cons, options={'disp': True, 'maxiter': 100, 'ftol': 1e-16})  
#     pca_list.append(res.x)
#     one_cov_matrix = np.matrix(subFrm.cov())
#     res2 = res.x*np.array(one_cov_matrix*np.matrix(res.x).T).T[0]
#     res3.append(res2)

# juzhen = np.array(pca_list)
# res4=np.array(res3)

# juzhen

# pca_list = []
# res3=[]
# frm_dts = list(df.index)
# weight0=[1/10]*10
# for dt in dts1:#dts为每季末
#     idx = frm_dts.index(dt)
#     subFrm = df.pct_change().fillna(0)[idx-80:idx]
#     res= minimize(pca_objective, weight0, args=subFrm, method='SLSQP',constraints=cons, options={'disp': True, 'maxiter': 100, 'ftol': 1e-16})  
#     pca_list.append(res.x)
#     one_cov_matrix = np.matrix(subFrm.cov())
#     res2 = res.x*np.array(one_cov_matrix*np.matrix(res.x).T).T[0]
#     res3.append(res2)

# juzhen = np.array(risk_parity_list)
# res4=np.array(res3)


# rc_dataframe=pd.DataFrame(res4[0:14],index=dts1,columns=df.columns)
# rc_dataframe.plot(kind='bar',figsize=(12,5))

pca_ret = cal_pnl(df2,pca_dict)

pca_ret.plot(title='主成分性风险平价',figsize=(12,5))

frmList = []
for i in pca_dict.keys():
    frmList.append(pd.DataFrame(pca_dict[i],index = df2.columns,columns = [i]))
weightfrm = pd.concat(frmList,axis=1).T

weightfrm.plot.area(title='主成分性风险平价',figsize = (12,5),colormap = 'Paired_r')



(pow(pca_ret.ix[len(pca_ret.index) - 1] / pca_ret.ix[0], 250/1000)-1)*100

