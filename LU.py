# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %% [markdown]
# author:leezeeyee   
# date:2021/5/11

# %%
import numpy as np 
import matplotlib.pyplot as plt
import scipy
import scipy.linalg
from datetime import datetime
from functools import wraps 

# %% [markdown]
# ## backward & forward substitution

# %%
def timer(func):
    @wraps(func)
    def layer(*args,**kwargs):
        #print(args)
        def count(*args,**kwargs):
            start=datetime.now()
            res=func(*args,**kwargs)
            time=(datetime.now()-start)
            return res,time
        return count(*args,**kwargs)
    return layer


# %%
@timer
def forward(L,b):
    d=np.shape(L)[0]
    Y=np.zeros(d)
    for i in range(d):
        l=L[i]
        Y[i]=b[i]
        for c in range(i):
            Y[i]-=Y[c]*l[c]
        Y[i]=Y[i]/l[i]
    return Y


# %%
@timer
# optimize
def forward_o(L,b):
    d=np.shape(L)[0]
    Y=np.copy(b)
    for i in range(d):
        l=L[i]
        Y[i]=Y[i]/l[i]
        Y[i+1:]-=Y[i]*L[i+1:,i]
    return Y


# %%
def test_forward():
    timel_o=[]
    timel=[]
    for d in range(50,800,50):
        
        M=np.random.rand(d,d)
        b=np.random.rand(d)
        P,L,U=scipy.linalg.lu(M)
        _,time=forward_o(L,b)
        # print(time.total_seconds())
        timel_o.append(time.total_seconds())
        _,time=forward(L,b)
        timel.append(time.total_seconds())
        # print(time.total_seconds())

    plt.plot(range(50,800,50),timel,label='origin')
    plt.plot(range(50,800,50),timel_o,label='optimize')
    plt.legend()
    plt.xlabel('matrix dimension')
    plt.ylabel('time cost')
    plt.title('forward substitution')
    plt.show()




# %%
@timer
def backward(U,b):
    d=np.shape(U)[0]
    Y=np.zeros(d)
    for i in range(d-1,-1,-1):
        #i=d-m-1
        l=U[i]
        Y[i]=b[i]
        for c in range(d-1,i,-1):
            Y[i]-=Y[c]*l[c]
        Y[i]=Y[i]/l[i]
    return Y


# %%
@timer
# optimize
def backward_o(L,b):
    d=np.shape(L)[0]
    Y=np.copy(b)
    for i in range(d-1,-1,-1):
        l=L[i]
        Y[i]=Y[i]/l[i]
        Y[:i]-=Y[i]*L[:i,i]
    return Y


# %%
def test_backward():

    timeb_o=[]
    timeb=[]
    for d in range(50,800,50):
        M=np.random.rand(d,d)
        b=np.random.rand(d)
        P,L,U=scipy.linalg.lu(M)
        res,time=backward_o(L,b)
        # print(res)
        # print(time.total_seconds())
        timeb_o.append(time.total_seconds())
        res,time=backward(L,b)
        # print(res)
        timeb.append(time.total_seconds())
        # print(time.total_seconds())
    plt.plot(range(50,800,50),timeb,label='origin')
    plt.plot(range(50,800,50),timeb_o,label='optimize')
    plt.legend()
    plt.xlabel('matrix dimension')
    plt.ylabel('time cost')
    plt.title('backward substitution')
    plt.show()

# %% [markdown]
# ## time consuming

# %%
@timer
def inv(L,b):
    return np.dot(np.linalg.inv(L),b)


# %%
def test_LU():
    d=4
    M=np.random.rand(d,d)
    b=np.random.rand(d)
    P,L,U=scipy.linalg.lu(M)

    pb=np.dot(np.linalg.inv(P),b)
    Y=forward.__wrapped__(L,pb)
    print(np.allclose(np.dot(L,Y),pb))
    X=backward.__wrapped__(U,Y)
    # print(np.dot(U,X))
    # print(((np.dot(M,X))-b))
    print(np.allclose(np.dot(M,X),b,rtol=0.1))

    # %% [markdown]
    # ## 置换矩阵
    # $P$为$LU$分解时的置换矩阵，为了防止因为主元为零（在数值计算中主元值很小）而交换矩阵两行的变换。如果我们想写出更普适的 $LU$ 分解式的话，必须把行交换情况考虑进去，即：$PA$ 先用行交换使得主元位置不为 0，行顺序正确。其后再用 $LU$ 分解。

    # %%
    print(np.dot(P,np.dot(L,U)))
    print(M)
    print(np.allclose(np.dot(P,np.dot(L,U)),M))


    # %%
    res=inv.__wrapped__(M,b)
    print(np.allclose(np.dot(M,res),b))
    print((np.dot(M,res)-b))

# %% [markdown]
# ## 精度低
# %% [markdown]
# 就是单位矩阵。上一章叙述了置换矩阵的性质，P-1 = PT，所以A = P-1LU = PTLU

# %%
@timer
def lu_s(M,b,optimize=True,trans=True):
    P,L,U=scipy.linalg.lu(M)
    if trans is True:
        b=np.dot(P.transpose(),b)
    else:    
        b=np.dot(np.linalg.inv(P),b)
    if optimize is False:
        Y=forward.__wrapped__(L,b)#np.dot(np.linalg.inv(L),b)
        X=backward.__wrapped__(U,Y)#np.dot(np.linalg.inv(U),Y)
    else:
        Y=forward_o.__wrapped__(L,b)#np.dot(np.linalg.inv(L),b)
        X=backward_o.__wrapped__(U,Y)#np.dot(np.linalg.inv(U),Y)
    return X


# %%
def time_compare():
    t1s=[]
    t2s=[]
    t3s=[]
    t4s=[]
    ranges=range(100,400,40)
    for d in ranges:
        # print('~~~~~~~~~~~~d='+str(d)+'~~~~~~~~~~~')
        #d=400
        M=np.random.rand(d,d)
        b=np.random.rand(d)
        # for i in range(3):
        res,t1=lu_s(M,b)
        res,t4=lu_s(M,b,trans=False)
        res,t2=lu_s(M,b,optimize=False,trans=False)
        res,t3=inv(M,b)
        t1s.append(t1.total_seconds())
        t2s.append(t2.total_seconds())
        t3s.append(t3.total_seconds())
        t4s.append(t4.total_seconds())


    # %%
    plt.plot(ranges,t1s,label='lu-optimize')
    plt.plot(ranges,t2s,label='lu-origin')
    plt.plot(ranges,t3s,label='inv')
    plt.plot(ranges,t4s,label='lu-notrans')
    plt.legend()
    plt.xlabel('matrix dimension')
    plt.ylabel('time cost')
    plt.title('lu VS inv')
    plt.show()


# %%
def test_LU_solve():
    d=290
    M=np.random.rand(d,d)
    b=np.random.rand(d)
    # for i in range(9):
    x1=lu_s.__wrapped__(M,b)
    x2=inv.__wrapped__(M,b)
    print(x1[:5])
    print(x2[:5])


# %%
if __name__=="__main__":
    test_backward()
    test_forward()
    test_LU()
    test_LU_solve()
    time_compare()




