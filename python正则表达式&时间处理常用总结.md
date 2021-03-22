# 正则表达式
## 正则
1. 所有正在表达式都在re模块中，导入该模块，使用'impor re'

### 常用总结
1. `\d`表示是一位数字字符，可以匹配0～9任意一个,若是想一次匹配3个数字则有两种方案：(1)`\d\d\d`,(2)`\d{3}`
2. `{n}`表示匹配前一个模式n变，也就是在一个模式后加上花括号包围的数字就等于匹配这个模式n词，比如👆的`\d{3}`
3. `re.compile()` 返回的是一个regex模式对象，创建模式对象后，可以调用内部方法,`正则模式对象.search('想查找字符串')`
4. `正则模式对象.search()` 若匹配到了，则返回匹配(Match)对象,对象结果调用.group()方法，返回具体匹配文本，若未匹配到，则返回None,注意`返回的字符串中第一次匹配的文本，若有多个，只返回第一个，若想返回所有，则用.findall()`
6. `匹配对象.group()` 若正则表达式内创建了分组（分组使用()分组)，则.group(n) 可以分组返回匹配文本，比如`r'(\d{3})-(\d{3}-\d{3})' 匹配'000-111-222' .group(1)返回000，group(2)返回111-222` 若n=0或者不穿入任何值，则返回整个匹配文本,注意: 第一个括号内，是第一个分组，第二个是第二个分组，若将以上正则改成`r'XXXX-(\d{3})-(\d{3}-\d{3})`,group(1) 和group(2) 返回结果一样
7. `匹配对象.groups()` 返回的是所有分组结果，返回类型是元组类型，可以按索引获取需要结果
8. `r'XXX1|XXX2'`,`|`管道的意思，匹配多个分组，也就是或的意思，匹配XXX1或者XXX2，⚠️：`调用.search() 和调用.findall()结果不同`
9. `r'(XXX1)?XXX2'`,`?`表示前面的分组(分组是()内的)在模式中可选的(可有可无的)，例：`r'aa(cc)?dd' 可以匹配aadd 也可以匹配aaccdd` `注意`：cc只出现0次/或者1次就不能匹配到aaccccdd
10. `r'(XXX1)*XXX2'`,`*`表示前面的分组(分组是()内的)在模式中可选的(可有可无的)，但可以匹配0次或者n次：例：`r'aa(cc)*dd' 可以匹配aadd 也可以匹配aaccdd，也可以匹配aaccccdd` 
11. `r'(XXX1)+XXX2'`,`+`表示前面的分组(分组是()内的)在模式中可选的(可有可无的)，但可以匹配1次或者n次：例：`r'aa(cc)*dd' 不能匹配aadd 可以匹配aaccdd，也可以匹配aaccccdd` 
12. `r'(XXX1){n}XXX2'`,`{n}`表示前面的分组(分组是()内的)在模式中可选的(可有可无的)，但只能匹配n次：例：`r'aa(cc){1}dd' 不能匹配aadd 可以匹配aaccdd，不能匹配aaccccdd` 
13. `r'(XXX1){n,m}XXX2'`,`{n,m}`表示前面的分组(分组是()内的)在模式中可选的(可有可无的)，但只能匹配>=n次and <=m：例：`r'aa(cc){1,2}dd' 不能匹配aadd 可以匹配aaccdd，可以匹配aaccccdd,不能匹配aaccccccdd` ,其中m要是不填写的画，就是`{n,}` 匹配至少n次
14. `正则模式对象.findall()` 返回的是所有匹配结果的字符串列表(和research()返回结果不一样),`若正则表达式中有分组，返回的是元组的列表，使用前测试下`
15. `通配记忆点` `\d`表示所有0-9任何数字,`\D`表示0-9意外所有的字符;`\w`表示任何字母，数字或者下划线等字符（非空白字符),`\W`表示非\w;`\s`表示任何空格，制表符或换行符（空白字符),`\S`表示非\s
16. `特定字符` `[]`表示可以匹配中括号内想特定匹配的字符集合,如`r[afgie] 'shawafnrebwg'匹配记过会返回(用findall()):['a','a','f','e','g']`
17. `区间(-)` `[0-9]` 表示0～9内所有数字，`r[0-9a-zA-Z]`表示所有数字及大小字母
18. `非特定字符` `[^特定字符]`表示 如：`r[^0-9a-zA-Z]`表示所有非数字及大小字母;
19. `r'^XXXX'` 表示必须以XXXX开始 ，如： `r'^cc' 可以匹配 “ccdd” 不能匹配“ddcc”`
20. `r'XXXX$'` 表示必须以XXXX结尾 ，如： `r'cc¥' 不能匹配 “ccdd” 能匹配“ddcc”`
21. `从头至尾巴都是数字的字符串匹配方式'` `r'^\d+$'`
22. `通配符.` `r'.cc'`可以匹配"1cc","dcc","ccc"," cc",".cc", '.'是可以匹配非换行之外所有的
23. `.*搭配使用` `r'(.*)'`匹配所有字符
24. `不区分大小写的匹配` `r'cc'`只能匹配到cc,若匹配到cc,cC,CC,Cc  则创建匹配模式对象时候增加`re.I`或者`re.IGNORECASE`更新为`re.compile(r'cc',re.I)` 
25. `.sub()方法`该方法可以实现将匹配的对象替换为新的文本 .sub(Par1,Par2),Par1参数是一个字符串，表示的是替换后新的文本,Par2参数表示的是正则匹配的字符串对象，返回的是替换完成后的字符串结果，也就是Par2被替换掉的结果
26. `复杂的正则表达式为了方便理解，可以加re.VERBOSE` 待补充




### 匹配url包含某个特定值的url，返回url结果
```python
## 需求：正则匹配，匹配日期为2018-03-20 的所有链接
url='https://sycm.taobao.com/bda/tradinganaly/overview/get_summary.json?dateRange=2018-03-20%7C2018-03-20&dateType=recent1&device=1&token=ff25b109b&_=1521595613462'
re_url=re.compile(r'(.*)dateRange=2018-03-20(.*)')
print(re_url.search(url).group(0))##https://sycm.taobao.com/bda/tradinganaly/overview/get_summary.json?dateRange=2018-03-20
print(re_url.search(url).group(1)) ##https://sycm.taobao.com/bda/tradinganaly/overview/get_summary.json?
print(re_url.search(url).group(2)) ##%7C2018-03-20&dateType=recent1&device=1&token=ff25b109b&_=1521595613462


```

### 从头至尾巴都是数字的字符串匹配方式
```python

import re
re_num=re.compile(r'^\d+$')
print(re_num.search('11111')) ##<re.Match object; span=(0, 5), match='11111'>
print(re_num.search('cc1111')) ##None
print(re_num.search('11111cc')) ##None
print(re_num.search('111cc11')) ##None

print(re_num.findall('11111')) #['11111']
print(re_num.findall('cc11111')) #[]
print(re_num.findall('11111cc')) #[]
print(re_num.findall('111cc11')) #[]

```




# 时间处理

## 两个常用模块
**time，datetime,一般time用于获取unix纪元时间戳，并加以处理，datetime则为美化后的日期格式 ** 数据分析常用的是pandas的时间模块


### 常用总结
1. `time.time()` 返回当前时刻的秒数，时间戳格式,浮点值类型，因为返回有小数点可以用round(time.time())，获取整形
2. `time.sleep(n)`  程序暂停类型，暂定n秒，
3. `datetime.datetime.strptime(str,'%Y-%m-%d %H:%M:%S')` 若想把字符串类型转化成datetime.datetime 类型，则可以使用该方法，但对str有要求， 若str='2020-01-03' 正format格式应为%Y-%m-%d，本例子中str必须是str='2011-01-03 12:00:01'
4. `pands.to_datetime()` date_str=['7/6/2011','8/6/2011'] 用该方法pd.to_datetime(date_str)，返回DatetimeIndex(['2011-07-06', '2011-08-06'], dtype='datetime64[ns]', freq=None) 


### 时间函数转换-时间戳转换
```python

#时间戳转换
import time
print(time.localtime()) ##time.struct_time(tm_year=2021, tm_mon=1, tm_mday=21, tm_hour=20, tm_min=33, tm_sec=4, tm_wday=3, tm_yday=21, tm_isdst=0)
print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())) ##2021-01-21 20:33:04
print(time.time()) ##1611232384.067116

now = int(time.time())
print(now) ##1611232474
timeArray = time.localtime(now)
print(timeArray) ##time.struct_time(tm_year=2021, tm_mon=1, tm_mday=21, tm_hour=20, tm_min=34, tm_sec=34, tm_wday=3, tm_yday=21, tm_isdst=0)
print(time.strftime("%Y-%m-%d %H:%M:%S", timeArray)) ## 2021-01-21 20:34:34



## 时间转换
import datetime
nowtime=datetime.datetime.now()
print(nowtime) ##2021-01-21 20:31:28.217652
print(nowtime.strftime("%Y-%m-%d %H:%M:%S")) ##2021-01-21 20:31:28
```

###  关于pands.to_datetime(str) 的使用说明
```python
##  若str=['7/6/2011','8/6/2011'] 或者str=['2011-07-06', '2011-08-06'],使用pd.to_datetime(str) 返回的结果DatetimeIndex(['2011-07-06', '2011-08-06'] 

## 若str=['2020-02-01 11:20:00', '2020-02-01 11:40:00', '2020-02-02 11:20:00', '2020-02-01 11:20:02','2020-02-01 11:22:00', '2020-02-01 11:30:00'],使用pd.to_datetime，效果如下

import pandas  as pd
str=['2020-02-01 11:20:00', '2020-02-01 11:40:00', '2020-02-02 11:20:00', '2020-02-01 11:20:02','2020-02-01 11:22:00', '2020-02-01 11:30:00']
print(pd.to_datetime(str))
'''
 返回 DatetimeIndex(['2020-02-01 11:20:00', '2020-02-01 11:40:00',
               '2020-02-02 11:20:00', '2020-02-01 11:20:02',
               '2020-02-01 11:22:00', '2020-02-01 11:30:00'],
              dtype='datetime64[ns]', freq=None)
    '''

print(pd.to_datetime(str,format='%Y-%m-%d'))
'''
返回 DatetimeIndex(['2020-02-01 11:20:00', '2020-02-01 11:40:00',
               '2020-02-02 11:20:00', '2020-02-01 11:20:02',
               '2020-02-01 11:22:00', '2020-02-01 11:30:00'],
              dtype='datetime64[ns]', freq=None)
'''



## 也就是原始str里到s的，使用to_datetime 肯定也到秒，只是转化了类型，此时若想仅提取日期的画，则需要把原始str切片获取到日的str，实现方案如下

## 若是pandas直接操纵，不需要pd.Series()   转换类型
print(pd.to_datetime(pd.Series(str).str.split(' ',expand=True)[0])) ##返回yyyy-mm-dd
print(pd.to_datetime(pd.Series(str).str.slice(start=0,stop=10)))##返回yyyy-mm-dd

## 方案2

str_split=[i.split(' ')[0] for i in str]
print(pd.to_datetime(str_split))##返回yyyy-mm-dd
print(pd.to_datetime(str_split,format='%Y-%m-%d'))##返回yyyy-mm-dd



```