#  SVD 图像降噪 ( image noise reduce )
## 原理
使用SVD分解图片信息矩阵，保留一部分奇异值然后重构对角矩阵，达到简单的图片降噪和压缩的目的。

## 环境依赖（environment）
- pillow or PIL 5.0.0 upper
- numpy 1.14.2 upper
- matplotlib 3.1.1 upper

## 测试运行（run）
- `python main.py [image path]`
``` bash

## 优化更新 
利用numpy的矩阵运算，将单次图片(750*500)处理压缩到1s内
❯ python .\main.py boy.jpeg
------init------
 time:
0:00:00.985616
------compress rate 10------
 time:
0:00:00.051863
(750, 500, 3)
------compress rate 30------
 time:
0:00:00.052857
(750, 500, 3)
------compress rate 50------
 time:
0:00:00.046874
(750, 500, 3)
------compress rate 70------
 time:
0:00:00.043886
(750, 500, 3)
------compress rate 90------
 time:
0:00:00.051864
(750, 500, 3)
```
## 效果总结（summary）
1.原始图片  

![image](test.jpg)

2.保留10%奇异值

![image](test_10%25.jpg)

2.保留50%奇异值  

![image](test_50%25.jpg)  

2.保留90%奇异值  

![image](test_90%25.jpg)


## 总结
保留的奇异值越多，图片的特征保留的越明显，当奇异值减少时，图片中的像素间的差距逐渐减小。
