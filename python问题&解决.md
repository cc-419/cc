1 升级现有的三方包命令
> jupyter页面下输入：!pip3 install --upgrade scipy,遇到超时error则输入：!pip3 --default-timeout=1000 install --upgrade scipy ；其中"!" 是在jupyter下必须要带的;终端下则不需要带
2 
> 输入命令：# -*- coding: UTF-8 -*-
3 设置随机种子
> import numpy as np np.random.seed(7654567)
4 查看安装的第三方包的版本
```python
import scipy
print(scipy.__version__) ##1.5.0
print(scipy.__file__) ##/opt/anaconda3/lib/python3.8/site-packages/scipy/__init__.py
```