```python
## 导入pandas
import pandas as pd
import numpy as np
## 查询pandas版本
print(pd.__version__)

## 创建Series 
arr=[0,1,2,3,4]
print('arr的类型是{},arr的值是{}'.format(type(arr),arr))
s1=pd.Series(arr)
print('s1的类型是{},s1的值是{}'.format(type(s1),s1))

## 使用Numpy创建Series
n = np.random.randn(5) ##创建一个随机 Ndarray 数组
print('n的类型是{},n的值是{}'.format(type(n),n))
index = ['a', 'b', 'c', 'd', 'e']
s2 = pd.Series(n, index=index)
print(s2)


## 使用字典类型创建Series
d = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}  # 定义示例字典
s3 = pd.Series(d) 
print(s3)

### 修改Series索引
print(s1)
s1.index=['A', 'B', 'C', 'D', 'E']
print(s1)

### Series拼接,append
s4=s3.append(s1)
print(s4)

### Series删除，s4删除e行,drop
print(s4)
s4.drop('e',inplace=True) ## ,dt.drop(labels=[列名称])
print(s4)


### Series修改特定元素
s4['A']=4
print(s4)

### Series四则运算
s4.add(s3) ## 按索引加和
s4.sub(s3) ## 按索引减
s4.mul(s3) ## 按索引乘
s4.div(s3) ## 按索引除
print(s4)

## 统计值 
s4.median() ## 中位数
s4.sum() ## 求和
s4.max()##最大值
s4.min()##最小值

##通过 NumPy 数组创建 DataFrame
dates = pd.date_range('today', periods=6)  # 定义时间序列作为 index
num_arr = np.random.randn(6, 4)  # 传入 numpy 随机数组
columns = ['A', 'B', 'C', 'D']  # 将列表作为列名
df1 = pd.DataFrame(num_arr, index=dates, columns=columns)
df1

## 字典创作DataFrame
data = {'animal': ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],
        'age': [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],
        'visits': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df2 = pd.DataFrame(data, index=labels)
df2

df2.dtypes ## 返回各字段类型
df2.head() ##返回前n行
df2.tail() ##返回后n行
df2.index ##df2的索引
df2.columns ## 返回df2的列名
df2.values ## 查看数据表的值
df2.describe() ## 数据表统计信息
df2.T ## 数据表转置
df2.sort_values(by='age')  # 按 age 升序排列 可以按照两个字段 by =[col1,col2],ascending=[True,False] 实现先升序后降序
df.drop_duplicates() ## 数据去重 ， 可以加keep参数，保留第一个重复值还是最后一个
df.dropna(how='all') ##删除整行/列都是缺失值的数据，若用how=‘any’表示删除的是只要行/列存在缺失值都删除
df.drop(columns=[],inplace=True,errors='ignore') ## 删除某字段,若遇到无该字段，则自动跳过（errors='ignore'）
df.drop(df[df[A]<0].index,errors='ignore') ## 按条件删除某行

df.fillna(0) ##df控制填充为0
df.fillna(method='pad') ## 使用缺失值前的值填充
df.fillna(method='bfill') ## 使用缺失值后的值填充
df.fillna(df.mean()['C':'E']) ## 使用C 列到 E 列的平均值填充
df.loc[df['A']=='0','A']=1 ## 更新特定列的值，比如将A列为0的值更新为1，一定要用loc,条件更新
data.loc[data['A']<2,'A']=2 ## 条件更新
df.nunique() ##去重计数
df.sort_index(ascending=True)  ###按照所以排序
df.isnull().any() ## 返回每个列（key）及对应是否有null值(bool值),若有，则为true，无则为False
pandas.DataFrame.sample(n=None,frac=None,replace=False,weights=None,random_state=None,axis=None) ## 抽样 n是样本量
df.query ## ??? 待补充

pd.get_dummies(groups) ### 字段dummy化
df = pd.DataFrame({'x': pd.Series(['1.0', '2.0', '3.0'], dtype=float), 'y': pd.Series(['1', '2', '3'], dtype=int)}) ##在python中如何创建包含不同类型数据的dataframe
df.reindex(index=(0,1,3,2))### pandas重新按照索引排序，排序前是0,1,2,3 排序后是0,1,3,2;
df.sort_index() ### 按照索引重新排序，比如上面是将3和2颠倒了，使用该方法后又重新排序为0，1，2，3
df.sort_values(by=2,ascending=False) ## 按照第2列，倒序排序，因为columns没有特定指
## 删除某列
del df['A'] ## 删除A列
df.pop(['A']) ## 删除A列
df.drop(['A'],axis=1,inplace=True) ## 原始数据删除A列,

## 特定列重命名
df.rename(columns={'a':'A','b':'B'})
## 全部列重命名
df.columns=['XXX','XXX',...] ## 数组长度为全部列的长度


## 统计数据集中没列字段NaN的个数 
df.isnull().sum() ## axis=0 表示按列，axis=1表示按行所以，返回每列/行NaN的个数

##DataFrame切片
df2[1:3] ##查询 2，3 行
df2['age'] ##选取age列
df2.age ##选取age列
df2[['age', 'animal']]  # 传入一个列名组成的列表，选取age，animal列
df2.iloc[1:3]  # 查询 2，3 行

# 生成 DataFrame 副本，方便数据集被多个不同流程使用
df3 = df2.copy()
df3
df3.isnull()  # 如果为空则返回为 True

## 添加列，修改值
num = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index=df3.index)
df3['No.'] = num  # 添加以 'No.' 为列名的新数据列
df3.iat[1, 1] = 2  # 索引序号从 0 开始，这里为 1, 1，# 修改第 2 行与第 2 列对应的值 3.0 → 2.0
df3.loc['f', 'age'] = 1.5 ##修改f行，age列为1.5

## DataFrame统计值计算
df3.mean()##返回各列平均值
df3['visits'].sum() ##返回visits列和

##将字符串转化为大小写转换
string = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca',
                    np.nan, 'CABA', 'dog', 'cat'])
print(string)
string.str.lower() ## 小写转换
string.str.upper() ##大写转换


## 缺失值处理
df4 = df3.copy() ## 复制数据集
print(df4)
df4.fillna(value=3) ## df4内空值军填充为3
df4.dropna(how='any') ### 任何存在 NaN 的行都将被删除

##DataFrame 按指定列对齐
left = pd.DataFrame({'key': ['foo1', 'foo2'], 'one': [1, 2]})
right = pd.DataFrame({'key': ['foo2', 'foo3'], 'two': [4, 5]})

print(left)
print(right)

# 按照 key 列对齐连接，只存在 foo2 相同，所以最后变成一行
pd.merge(left, right, on='key',how='innner join') ## 相当于 sql join  left表的key列=right的key left join right on left.key=right.key

##DataFrame 文件操作
df3.to_csv('animal.csv') ##文件写入
df_animal = pd.read_csv('animal.csv') ## 文件读取
df3.to_excel('animal.xlsx', sheet_name='Sheet1') ##文件写入，excel文件，sheet1名称
pd.read_excel('animal.xlsx', 'Sheet1', index_col=None, na_values=['NA']) ##文件读取

##时间序列索引
dti = pd.date_range(start='2018-01-01', end='2018-12-31', freq='D') ## 生成从18年整，按天为间距，freq='nD' n可以为任何数值,默认为1
s = pd.Series(np.random.rand(len(dti)), index=dti)
s[s.index.weekday == 2].sum() ## 统计每周三数据只和
s.resample('M').mean() ##按月统计平均值

## 将 Series 中的时间进行转换（秒转分钟）
s = pd.date_range('today', periods=100, freq='S')## 生成从今天，按s为间距，freq='nS',n可以为任何数值,默认为1
ts = pd.Series(np.random.randint(0, 500, len(s)), index=s)
ts.resample('Min').sum() ## 按秒之和，

## UTC 世界时间标准
s = pd.date_range('today', periods=1, freq='D')  # 获取当前时间
ts = pd.Series(np.random.randn(len(s)), s)  # 随机数值
ts_utc = ts.tz_localize('UTC')  # 转换为 UTC 时间
ts_utc

##不同时间表示方式的转换，还需要加深记忆
rng = pd.date_range('1/1/2018', periods=5, freq='M') ## x
ts = pd.Series(np.random.randn(len(rng)), index=rng)
print(ts)
ps = ts.to_period() ##
print(ps)
ps.to_timestamp()

## 填充，线性填充
# 生成一个 DataFrame
df = pd.DataFrame({'A': [1.1, 2.2, np.nan, 4.5, 5.7, 6.9],
                   'B': [.21, np.nan, np.nan, 3.1, 11.7, 13.2]})
df_interpolate = df.interpolate() ##线性插值填充,支持method参数，如果你的数据增长速率越来越快，可以选择 method='quadratic'二次插值；如果数据集呈现出累计分布的样子，推荐选择 method='pchip'。如果需要填补缺省值，以平滑绘图为目标，推荐选择 method='akima'。
df_interpolate

##数据透视表 pivot_table
df = pd.DataFrame({'A': ['one', 'one', 'two', 'three'] * 3,
                   'B': ['A', 'B', 'C'] * 4,
                   'C': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
                   'D': np.random.randn(12),
                   'E': np.random.randn(12)})

print(df)

pd.pivot_table(df, index=['A', 'B'])
#A,B为行，对余下数值类型字段做均值处理，aggfunc默认为mean，因为C列非数值类型，且未被选为列，所以数据结果展示的是，A&B标签组合，D和E的均值
## 以上方法等同于 pd.pivot_table(df, index=['A', 'B'],values=['D','E'])
pd.pivot_table(df,index=df['A'],columns=df['B'],aggfunc='mean',values=['D','E'])
## 对df数据表，按行为A列，表头为B列，对数据D列数据做均值处理，类似excel数据透视表，A字段为行，B字段为列，D,E字段为值，方法是秋均值
pd.pivot_table(df, index=['A', 'B'],values=['D','E'],aggfunc=[np.sum,len]) ## aggfunc可以list

pd.pivot_table(df, values=['D'], index=['A', 'B'],
               columns=['C'], aggfunc=np.sum, fill_value=0) ## 在透视表中由于不同的聚合方式，相应缺少的组合将为缺省值，可以加入 fill_value 对缺省值处理。
### 数据透视表之后一般要把索引重置
df_tmp.reset_index() ## 对数据透视表结果的数据df_tmp进行索引重置，会生成新的索引


## set_categories
df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "raw_grade": [
                  'a', 'b', 'b', 'a', 'a', 'e']})
print(type(df['raw_grade']))
df["grade"] = df["raw_grade"].astype("category")
df["grade"].cat.categories = ["very good", "good", "very bad"] ## 对grade类型重命名，a会为Very good b：good，e：very bad
df["grade"] = df["grade"].cat.set_categories(
    ["very bad", "bad", "medium", "good", "very good"]) ## 自行试验
df.groupby("grade").size() ## 查看各字段按grade为维度计数的结果
df.groupby("grade").mean() ## 查看各字段按grade为维度均值的结果
df.groupby("grade")[列名].nunique() ### 按grade分组，查看特定字段去重值，.sum()/.mean() 都可以实现
df.groupby("grade").agg({'列1':np.size,'列2':np.sum}) ### 按grade分组，查看列1，计数，列2，求和

## 缺失值拟合
df = pd.DataFrame({'From_To': ['LoNDon_paris', 'MAdrid_miLAN', 'londON_StockhOlm',
                               'Budapest_PaRis', 'Brussels_londOn'],
                   'FlightNumber': [10045, np.nan, 10065, np.nan, 10085],
                   'RecentDelays': [[23, 47], [], [24, 43, 87], [13], [67, 32]],
                   'Airline': ['KLM(!)', '<Air France> (12)', '(British Airways. )',
                               '12. Air France', '"Swiss Air"']})
df['FlightNumber'] = df['FlightNumber'].interpolate().astype(int) ## interpolate 为缺失值拟合结果，在FilghtNumber中有数值缺失，其中数值为按 10 增长，补充相应的缺省值使得数据完整，并让数据为 int 类型
df

## 数据列拆分 From_To
temp = df.From_To.str.split('_', expand=True)##将From_to依照_拆分为独立两列建立为一个新表
temp.columns = ['From', 'To']
temp['From'] = temp['From'].str.capitalize() ## 数字标准化，londON更新为London
temp['To'] = temp['To'].str.capitalize()
df = df.drop('From_To', axis=1) ## 删除原油的From_To列
df = df.join(temp) ## 将temp 合并到df数据表中
df['Airline'] = df['Airline'].str.extract(
    '([a-zA-Z\s]+)', expand=False).str.strip() ## Airline列有好多数值/其它字符，该操作为近提取字母的值，并
delays = df['RecentDelays'].apply(pd.Series) ## 作用是将RecentDelays列按,进行拆飞了
delays.columns = ['delay_{}'.format(n)  for n in range(1, len(delays.columns)+1)] #对columns进行重命名
df = df.drop('RecentDelays', axis=1).join(delays) ## drop原油数据，并join处理后的数据表

## 数据区间划分，使用了lambda
df=pd.DataFrame({'name':['Alice','Bob','Candy','Dany','Ella','Frank','Grace','Jenny'],'grades':[58,83,79,65,93,45,61,88]})
def choice(x):
    if x > 60:
        return 1
    else:
        return 0
df.grades = pd.Series(map(lambda x: choice(x), df.grades))
## 重命名操作
new_name={'Alice':'cc','BOb':'bod'}
df['new_name']=df['name'].map)(new_name) ## 未匹配上的会被职位空

df=pd.DataFrame({'name':['Alice','Bob','Candy','Dany','Ella','Frank','Grace','Jenny'],'grades':[58,83,79,65,93,45,61,88]})
def choice(x):
    if x > 60:
        return 1
    else:
        return 0
df.grades = pd.Series(map(lambda x: choice(x), df.grades))
## 替换某列某个特定值，如下将name未Alice替换的name值替换为cc
df.loc[df['name']=='Alice','name']='cc'
df


##不是很懂
df = pd.DataFrame({'A': [1, 2, 2, 3, 4, 5, 5, 5, 6, 7, 7]})
df.loc[df['A'].shift() != df['A']] ##尝试将 A 列中连续重复的数据清除

##数据归一化，用最大最小值归一
## 定义方法，直接对DataFrame进行归一化处理，返回归一化后的DataFrame
def normalization(df):
    numerator = df.sub(df.min())
    denominator = (df.max()).sub(df.min()) ## 分母，最大值减最小值
    Y = numerator.div(denominator)
    return Y


df = pd.DataFrame(np.random.random(size=(5, 3)))
print(df)
normalization(df)


### 数据可视化,pandas自带的可视化方法
%matplotlib inline
ts = pd.Series(np.random.randn(100), index=pd.date_range('today', periods=100))
ts = ts.cumsum() ## 返回累积值
ts.plot()

df = pd.DataFrame(np.random.randn(100, 4), index=ts.index,
                  columns=['A', 'B', 'C', 'D'])
df = df.cumsum() ## 累积值，各列累积
df.plot() ##折线图

df = pd.DataFrame({"xs": [1, 5, 2, 8, 1], "ys": [4, 2, 1, 9, 6]})
df = df.cumsum()
df.plot.scatter("xs", "ys", color='red', marker="*") ## 散点图

df = pd.DataFrame({"revenue": [57, 68, 63, 71, 72, 90, 80, 62, 59, 51, 47, 52],
                   "advertising": [2.1, 1.9, 2.7, 3.0, 3.6, 3.2, 2.7, 2.4, 1.8, 1.6, 1.3, 1.9],
                   "month": range(12)
                   })

ax = df.plot.bar("month", "revenue", color="yellow") ## 画出了X轴为month，y轴为revenue的柱状图
df.plot("month", "advertising", secondary_y=True,ax=ax) ## 上图中有增加了X轴为month，次坐标轴(secondary_y=True)的折线图,其中ax设置为ax才会在一个图中展示，若无该参数设置，则不会展示在一个图中

```

### 缺失值处理
```python 
### 缺失值行/列删除
data.dropna(how='all',inplace=True) ##删除整行/列都是缺失值的数据，若要删除只要存在缺失值的行/列，则用户how='any'
### 缺失值填充
data['缺失值字段名'].fillna(data['缺失值字段名'].mean(),inplace=True) ## 用均值填充
##相邻值填充
data['缺失值字段名'].fillna(data[method='bfill',inplace=True)
## 删除某个值为null的行
data.dropna(subset=['缺失值字段名'],inplace=True) ##删除"缺失值字段名"为空的整行记录
```


### 数据划分区间
```python 
## 划分区间 [32,50),[50,80),....[120,316) right参数设置是否右开，precision设置上线阈值的精度
pd.cut(data_c4['area_1'],bins=[32,50,80,100,120,316],right=False,precision=0,include_lowest=True)

## 方法2，自己些函数来实现 

def funname_cc(X):
    if 条件1:
        XXX
    elif 条件2：
        XXX
    else：
        XXX
data_c4['area_1']=data_c4['area'].map(funname_cc)
```

### 实现sql的union 
```python
pd.concat([data1,data2])  ## union
pd.concat([order_data, order_data2]).drop_duplicates() ## union all

```

### 实现sql的字符串的截取 substr()
```python
data['col1'].str[0:10] ##截取前 
data['col1'].astype(str).str[0:10] ##截取前 
``` 

### 实现sql的模糊匹配, sql like '%SSS%' like 'SSS%'

```python
data['col1'].astype(str).str.contains('08-01')]
data['col1'].astype(str).str.startswith('08-01')] 
data['col1'].astype(str).str.endswith('08-01')] 
data['col1'].astype(str).str.extract('08-01')]  ## 正则
```

### python：replace sql：regexp_replace


### pd.to_datetime(ts, format='%Y-%m-%d %H:%M:%S')



### 画图中文显示问题
```python
plt.rcParams['font.family']='sans-serif' 
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False 

```

### sql A in ('1','2') pandas.isin([1,2])
```python
pd['A'].isin([1,2])

```

### sql lag(字段,1) 实现取前一行数据
```python
pd['A'].shift(1)

```


### apply 做映射处理
```python
def key_2_newkey(x):
    if x=='No Policy':
        return 0
    elif x=='flexible':
        return 1
    elif x=='moderate':
        return 2
    else:
        return 3
        
data['dim_listing_cancellation_policy_2']=data['dim_listing_cancellation_policy'].apply(key_2_newkey)

```
### lambda的使用
```python
data['dim_has_ticket'] = data['m_tickets'].apply(lambda x: 1 if x >0 else 0) 
```