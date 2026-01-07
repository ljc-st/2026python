import copy
def use_list_copy():
    #使用列表自身的copy
    a=[1,2,3]
    b=a.copy()# a保留原有部分不变
    b[0]=10
    print(a)
    print(b)
def use_copy():
    #使用import的copy
    c=[1,2,3]
    d=copy.copy(c)
    d[0]=10
    print(c)
    print(d)
def use_copy2():
    #浅copy
    a=[1,2]
    b=[3,4]
    c=[a,b]
    d=copy.copy(c)
    print(id(c))
    print(id(d))
    a[0]=10
    print(id(c[0]),id(c[1]))
    print(id(d[0]), id(d[1]))
    print(a,b)
def use_deepcopy():
    #深copy
    a=[1,2]
    b=[3,4]
    c=[a,b]
    d=copy.deepcopy(c)
use_list_copy()
use_copy()
use_copy2()
use_deepcopy()