# %%
import time

from sklearn.datasets import load_iris, fetch_20newsgroups, fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
# %% [markdown]
# load直接加载的内存的，数据集比较小，并不会保存到本地磁盘
# fetch数据集比较大，下载下来后会存在本地磁盘，下一次就不会再连接sklearn的服务器
# 
# %%
#鸢尾花数据集，查看特征，目标，样本量

li = load_iris()

print("获取特征值")
print(type(li.data))
print('-' * 50)
print(li.data.shape) # 150个样本，4个特征,一般看shape
li.data
# %%
print("目标值")
print(li.target)
print('-' * 50)
print(li.DESCR)
print('-' * 50)
print(li.feature_names)  # 重点,特征名字
print('-' * 50)
print(li.target_names) # 目标名字
# %%
print(li.data.shape)
li.target.shape
# %%
# 注意返回值, 训练集 train  x_train, y_train        测试集  test   x_test, y_test，顺序千万别搞错了
# 默认是乱序的,random_state为了确保两次的随机策略一致，就会得到相同的随机数据，往往会带上
x_train, x_test, y_train, y_test = train_test_split(li.data, li.target, test_size=0.25, random_state=1)

# print("训练集特征值和目标值：", x_train, y_train)
print("训练集特征值shape", x_train.shape)
print('-'*50)
# print("测试集特征值和目标值：", x_test, y_test)
print("测试集特征值shape", x_test.shape)
# %%
150*0.25
# %%
# 下面是比较大的数据，需要下载一会，20类新闻
#subset代表下载的数据集类型，默认是train，只有训练集
news = fetch_20newsgroups(subset='all', data_home='data')
# print(news.feature_names)  #这个数据集是没有的，因为没有特征，只有文本数据
# print(news.DESCR)
print('第一个样本')
print(news.data[0])


# %%
print('特征类型')
print(type(news.data))
print('-' * 50)
print(news.target[0:15])
from pprint import pprint
pprint(list(news.target_names))
# %%
len(news.target_names)
# %%
print('-' * 50)
print(len(news.data))
print('新闻所有的标签')
print(news.target)
print('-' * 50)
print(min(news.target), max(news.target))
# %%
#因为新版本sklearn去掉了 这个数据集，不再讲解
# 接着来看回归的数据,是波士顿房价
# lb = load_boston()
#
# print("获取特征值")
# print(lb.data[0])  #第一个样本特征值
# print(lb.data.shape)
# print('-' * 50)
# print("目标值")
# print(lb.target)
# print('-' * 50)
# print(lb.DESCR)
# print('-' * 50)
# print(lb.feature_names)
# print('-' * 50)
# 回归问题没这个,打印这个会报错
# print(lb.target_names)
# %%
house=fetch_california_housing(data_home='data')
print("获取特征值")
print(house.data[0])  #第一个样本特征值
print('样本的形状')
print(house.data.shape)
print('-' * 50)

# %%
print("目标值")
print(house.target[0:10])
print('-' * 50)
print(house.DESCR)
print('-' * 50)
print(house.feature_names)
print('-' * 50)
# %% [markdown]
# # 2 分类估计器
# %%
np.sqrt(15*15+14*14)
# %%
# K近邻
"""
K-近邻预测用户签到位置
:return:None
"""
# 读取数据
data = pd.read_csv("./data/FBlocation/train.csv")

print(data.head(10))
print(data.shape)
print(data.info())
# 处理数据
# 1、缩小数据,查询数据,为了减少计算时间
data = data.query("x > 1.0 &  x < 1.25 & y > 2.5 & y < 2.75")


# %%
data.shape
# %%
data.describe()
# %%
# 处理时间的数据
time_value = pd.to_datetime(data['time'], unit='s')

print(time_value.head(10))  #最大时间是1月10号
# %%
# 把日期格式转换成 字典格式，把年，月，日，时，分，秒转换为字典格式，
time_value = pd.DatetimeIndex(time_value)
#
print('-' * 50)
print(time_value[0:10])
# %%
data.shape
# %%
print('-' * 50)
# 构造一些特征，执行的警告是因为我们的操作是复制，loc是直接放入
print(type(data))
# data['day'] = time_value.day
# data['hour'] = time_value.hour
# data['weekday'] = time_value.weekday
#日期，是否是周末，小时对于个人行为的影响是较大的(例如吃饭时间去饭店，看电影时间去电影院等),所以才做下面的处理
data.insert(data.shape[1], 'day', time_value.day) #data.shape[1]是代表插入到最后的意思,一个月的哪一天
data.insert(data.shape[1], 'hour', time_value.hour)#是否去一个地方打卡，早上，中午，晚上是有影响的
data.insert(data.shape[1], 'weekday', time_value.weekday) #0代表周一，6代表周日，星期几

#
# 把时间戳特征删除
data = data.drop(['time'], axis=1)
print('-' * 50)
data.head()
# %%
#星期天，实际weekday的值是6
per = pd.Period('1970-01-01 18:00', 'h')
per.weekday
# %%
#观察数据，看下是否有空值，异常值
data.describe()
# %%
# # 把签到数量少于n个目标位置删除，place_id是标签，即目标值
place_count = data.groupby('place_id').count()
place_count
# %%
place_count['x'].describe() #打卡地点总计805个，50%打卡小于2次
# %%
# # 把index变为0,1,2，3,4,5,6这种效果，从零开始排，原来的index是row_id
#只选择去的人大于3的数据，认为1,2,3的是噪音，这个地方去的人很少，不用推荐给其他人
tf = place_count[place_count.row_id > 3].reset_index()
tf  #剩余的签到地点
# %%
# 根据设定的地点目标值，对原本的样本进行过滤
#isin可以过滤某一列要在一组值
data = data[data['place_id'].isin(tf.place_id)]
data.shape
# %%
# # 取出数据当中的特征值和目标值
y = data['place_id']
# 删除目标值，保留特征值，
x = data.drop(['place_id'], axis=1)
# 删除无用的特征值，row_id是索引,这就是噪音
x = x.drop(['row_id'], axis=1)
print(x.shape)
print(x.columns)
# %% [markdown]
# # 上面预处理完成
# %%
# li = load_iris()
# x,y=li.data,li.target
# %%
# 进行数据的分割训练集合测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

# 特征工程（标准化）,下面3行注释，一开始我们不进行标准化，看下效果，目标值要不要标准化？
std = StandardScaler()
# #
# # # 对测试集和训练集的特征值进行标准化,服务于knn fit
x_train = std.fit_transform(x_train)
# # transform返回的是copy，不在原有的输入对象中去修改
# print(id(x_test))
print(std.mean_)
print(std.var_)
x_test = std.transform(x_test)  #transfrom不再进行均值和方差的计算，是在原有的基础上去标准化
print('-' * 50)
# print(id(x_test))
print(std.mean_)
print(std.var_)
# %%
x_train.shape
# %%
# # 进行算法流程 # 超参数，可以通过设置n_neighbors=5，来调整结果好坏
knn = KNeighborsClassifier(n_neighbors=6)

# # fit， predict,score，训练，knn的fit是不训练的，只是把训练集的特征值和目标值放入到内存中
knn.fit(x_train, y_train)
# # #
# # # 得出预测结果
y_predict = knn.predict(x_test)
# #
print("预测的目标签到位置为：", y_predict[0:10])
# # #
# # # # 得出准确率,是评估指标
print("预测的准确率:", knn.score(x_test, y_test))
# print(y_predict)
# y_test
# %%
print(max(time_value))
# %% [markdown]
# # 调超参的方法，网格搜索
# %%
#网格搜索时讲解
# # 构造一些参数（超参）的值进行搜索
param = {"n_neighbors": [3, 5, 10, 12, 15],'weights':['uniform', 'distance']}
#
# 进行网格搜索，cv=3是3折交叉验证，用其中2折训练，1折验证
gc = GridSearchCV(knn, param_grid=param, cv=3)

gc.fit(x_train, y_train)  #你给它的x_train，它又分为训练集，验证集

# 预测准确率，为了给大家看看
print("在测试集上准确率：", gc.score(x_test, y_test))

print("在交叉验证当中最好的结果：", gc.best_score_) #最好的结果

print("选择最好的模型是：", gc.best_estimator_) #最好的模型,告诉你用了哪些参数

print("每个超参数每次交叉验证的结果：")
gc.cv_results_
# %%
"""
朴素贝叶斯进行文本分类
:return: None
"""
news = fetch_20newsgroups(subset='all', data_home='data')

print(len(news.data))  #样本数，包含的特征
print('-'*50)
print(news.data[0]) #第一个样本 特征
print('-'*50)
print(news.target) #标签
print(np.unique(news.target)) #标签的类别
print(news.target_names) #标签的名字
# %%
print('-'*50)
# 进行数据分割
x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=1)

# 对数据集进行特征抽取
tf = TfidfVectorizer()

# 以训练集当中的词的列表进行每篇文章重要性统计['a','b','c','d']
x_train = tf.fit_transform(x_train)
#针对特征内容，可以自行打印，下面的打印可以得到特征数目，总计有15万特征
print(len(tf.get_feature_names_out()))
# %%
print(tf.get_feature_names_out()[100000])
# %%
print(tf.get_feature_names_out()[0:10])
# %%
import time
# 进行朴素贝叶斯算法的预测,alpha是拉普拉斯平滑系数，分子和分母加上一个系数，分母加alpha*特征词数目
mlt = MultinomialNB(alpha=1.0)

# print(x_train.toarray())
# 训练
start=time.time()
mlt.fit(x_train, y_train)
end=time.time()
end-start #统计训练时间
# %%
x_transform_test = tf.transform(x_test)  #特征数目不发生改变
print(len(tf.get_feature_names_out())) #查看特征数目
# %%
start=time.time()
y_predict = mlt.predict(x_transform_test)

print("预测的前面10篇文章类别为：", y_predict[0:10])

# 得出准确率,这个是很难提高准确率，为什么呢？
print("准确率为：", mlt.score(x_transform_test, y_test))
end=time.time()
end-start
# %%
#预测的文章数目
len(y_predict)
# %%
# 目前这个场景我们不需要召回率，support是真实的为那个类别的有多少个样本
print(classification_report(y_test, y_predict,
      target_names=news.target_names))

# %%
y_test.shape #测试集中有多少 样本
# %%
y_test1 = np.where(y_test == 0, 1, 0)
print(y_test1.sum()) #label为0的样本数
# %%
y_predict1 = np.where(y_predict == 0, 1, 0)
print(y_predict1.sum())
# %%
(y_test1*y_predict1).sum()
# %%
153/168
# %%
153/199
# %%
max(y_test),min(y_test)
# %%
# 把0-19总计20个分类，变为0和1
# 5是可以改为0到19的
y_test1 = np.where(y_test == 5, 1, 0)
print(y_test1.sum()) #label为5的样本数
y_predict1 = np.where(y_predict == 5, 1, 0)
print(y_predict1.sum())
# roc_auc_score的y_test只能是二分类,针对多分类如何计算AUC
print("AUC指标：", roc_auc_score(y_test1, y_predict1))
# %%
y_test1,y_predict1
# %%
#算多分类的精确率，召回率，F1-score
FP=np.where((np.array(y_test1)-np.array(y_predict1))==-1,1,0).sum()   #FP是18
TP=y_predict1.sum()-FP #TP是196
print(TP)
FN=np.where((np.array(y_test1)-np.array(y_predict1))==1,1,0).sum() #FN是34
print(FN)#FN是1
TN=np.where(y_test1==0,1,0).sum()-FP  #4464
print(TN)
# %%
TP/(TP+FP) #精确率
# %%
TP/(TP+FN)  #召回率
# %%
#F1-score
2*TP/(2*TP+FP+FN)
# %% [markdown]
# 
# %%
del news
del x_train
del x_test
del y_test
del y_predict
del tf
# %% [markdown]
# # 3 决策树
# %%
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
# %%
np.log2(1/32)
# %%
1 / 2 * np.log2(1 /2) + 1 / 2 * np.log2(1 /2)
# %%
1 / 3 * np.log2(1 / 3) + 2 / 3 * np.log2(2 / 3)
# %%
0.01 * np.log2(0.01) + 0.99 * np.log2(0.99)
# %%
"""
决策树对泰坦尼克号进行预测生死
:return: None
"""
# 获取数据
titan = pd.read_csv("./data/titanic.txt")
titan.info()
# %%
# 处理数据，找出特征值和目标值
x = titan[['pclass', 'age', 'sex']]

y = titan['survived']
print(x.info())  # 用来判断是否有空值
x.describe(include='all')
# %%
x.loc[:,'age'].max()
# %%
# 一定要进行缺失值处理,填为均值
mean=x['age'].mean()
x.loc[:,'age']=x.loc[:,'age'].fillna(mean)

# %%
x.info()
# %%


# 分割数据集到训练集合测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=4)
print(x_train.head())
# %%
type(x_train)
# %%
sum(y_train)
# %%
#性别是女性的数量
x_train[x_train['sex'] == 'female'].count()
# %%
y_train
# %%
#女性中存活的情况对比
z=x_train.copy() #z是为了把特征和目标存储到一起
z['survived'] = y_train #把目标值存储到z中
z[z['sex'] == 'female']['survived'].value_counts() #男性中存活的情况
# %%
y_train.value_counts() #没存活的是650，存活的是334
# %%
x_train.loc[:,'sex'].value_counts()
# %%
230/(230+111)
# %%
#查看未存活的人的数量
x_train
# %%
x_train.to_dict(orient="records") #把df变为字典，样本变为一个一个的字典，字典中列名变为键
# %%
# 进行处理（特征工程）特征-》类别-》one_hot编码
dict = DictVectorizer(sparse=False)

# 这一步是对字典进行特征抽取,to_dict可以把df变为字典，records代表列名变为键
x_train = dict.fit_transform(x_train.to_dict(orient="records"))
print(type(x_train))
print(dict.get_feature_names_out())
print('-' * 50)
x_test = dict.transform(x_test.to_dict(orient="records"))
print(x_train)
# %%
# 用决策树进行预测，修改max_depth试试,修改criterion为entropy
#树过于复杂，就会产生过拟合
dec = DecisionTreeClassifier()

#训练
dec.fit(x_train, y_train)

# 预测准确率
print("预测的准确率：", dec.score(x_test, y_test))

# 导出决策树的结构
export_graphviz(dec, out_file="tree.dot",
                feature_names=['age', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', 'female', 'male'])

# %%
#调整决策树的参数
# 分割数据集到训练集合测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=4)
# 进行处理（特征工程）特征-》类别-》one_hot编码
dict = DictVectorizer(sparse=False)

# 这一步是对字典进行特征抽取
x_train = dict.fit_transform(x_train.to_dict(orient="records"))
x_test = dict.transform(x_test.to_dict(orient="records"))

# print(x_train)
# # 用决策树进行预测，修改max_depth为10，发现提升了,min_impurity_decrease带来的增益要大于0.01才会进行划分
dec = DecisionTreeClassifier(max_depth=7,min_impurity_decrease=0.01,min_samples_split=20)

dec.fit(x_train, y_train)
#
# # 预测准确率
print("预测的准确率：", dec.score(x_test, y_test))
#
# # 导出决策树的结构
export_graphviz(dec, out_file="tree1.dot",
                feature_names=dict.get_feature_names_out())
# %%
y_train.shape
# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=4)
# 进行处理（特征工程）特征-》类别-》one_hot编码
dict = DictVectorizer(sparse=False)

# 这一步是对字典进行特征抽取
x_train = dict.fit_transform(x_train.to_dict(orient="records"))
x_test = dict.transform(x_test.to_dict(orient="records"))
# %%
# 随机森林进行预测 （超参数调优），n_jobs充分利用多核的一个参数
rf = RandomForestClassifier(n_jobs=-1)
# 120, 200, 300, 500, 800, 1200,n_estimators森林中决策树的数目，也就是分类器的数目
# max_samples  是最大样本数
#bagging类型
param = {"n_estimators": [1500,2000, 5000], "max_depth": [2, 3, 5, 8, 15, 25]}

# 网格搜索与交叉验证
gc = GridSearchCV(rf, param_grid=param, cv=3)

gc.fit(x_train, y_train)

print("准确率：", gc.score(x_test, y_test))

print("查看选择的参数模型：", gc.best_params_)

print("选择最好的模型是：", gc.best_estimator_)

# print("每个超参数每次交叉验证的结果：", gc.cv_results_)

