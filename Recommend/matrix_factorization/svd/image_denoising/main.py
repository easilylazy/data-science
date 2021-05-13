# %%
import numpy as np
from PIL import Image
import sys
from datetime import datetime
import matplotlib.pyplot as plt

# %%
class SVD_compress():
    def __init__(self,data):
        self.data=data
        self.u=[]
        self.sigma=[]
        self.v=[]
        # scale 代表你要保留的奇异值比例
        for i in range(3):
            u,sigma,v = np.linalg.svd(data[:,:,i])
            self.u.append(u)
            self.sigma.append(sigma)
            self.v.append(v)
        

    # 使用奇异值总和的百分比进行筛选
    def svd(self,rgb,scale):

        svd_data = np.zeros(self.data[:,:,rgb].shape)
        u=np.copy(self.u[rgb])
        sigma=np.copy(self.sigma[rgb])
        v=np.copy(self.v[rgb])
        total = sum(sigma)
        sum_data = 0
        for index,item in enumerate(sigma):  
            # svd_data += item * np.dot(u[:,index].reshape(-1,1),v[index,:].reshape(1,-1))
            sum_data += item
            if sum_data >= scale * total:
                break
        # print(u.shape,sigma.shape,v.shape)
        # print(len(sigma))
        sigma[index:]=0
        # print(len(sigma))
        if len(u[0])>len(sigma):
            svd_data = np.dot(u[:,:len(sigma)]*sigma,v)
        else:
            svd_data = np.dot(u*sigma,v[:len(sigma)])
        return svd_data

    def compress(self,scale):
        r = self.svd(0,scale)
        g = self.svd(1,scale)
        b = self.svd(2,scale)
        # print(r.shape)

        result = np.stack((r,g,b),2)
        result[result > 255] = 255
        result[result < 0] = 0
        result = result.astype(int)
        return result

# %%
def main():
    if len(sys.argv)>=2:
        img_path=sys.argv[1]
    else:
        img_path='test.jpg'
    # img_path='C:\\Users\\cascara\\Pictures\\test\\boy.jpeg'
    try:
        image = Image.open(img_path)
    except:
        raise Exception('no such file exists')

    try:
        title=(img_path.split('.')[-2])
        title=(title.split('\\')[-1])

    except:
        print('no suffix')
        title='untitled'
    arr=np.asarray(image)
    time=datetime.now()
    Compress=SVD_compress(arr)
    deltaTime=(datetime.now()-time)
    print('------init------\n time: ')
    print(deltaTime)
    # 原生 range 不支持浮点数，所以用 np.arange 代替
    for c in np.arange(.1,.9,.2):
        time=datetime.now()
        result = Compress.compress(c)
        deltaTime=(datetime.now()-time)
        print('------compress rate '+str(int(100 * c))+'------\n time: ')
        print(deltaTime)
        print(result.shape)
        proImage=Image.fromarray(np.uint8(result))
        proImage.save('res/'+title+str(int(100 * c))+'%.jpg')



    plt.figure("img")
    plt.imshow(result)
    plt.show()
if __name__ == '__main__':
    main()

