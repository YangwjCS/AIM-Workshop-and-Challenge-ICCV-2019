Example-based-Bokeh-Effect-Challenge

Requirement
Python 3.7.3
torch 0.4.1
scipy 1.3.1
numpy 1.16.4
imageio 2.5.0
cv 2 4.1.0

dataset download
train dataset：https://pan.baidu.com/s/1iB4N93L_SFRAoEHEm4pIOQ
test  dataset：https://pan.baidu.com/s/1POpip1m0JDiQU4wodoCnMg

训练代码:
1.首先先修改路径，然后在dataset/data.py中修改图片大小，将一个修改为resize(256*256),
另一个修改为crop(512*512)分别运行python train0.py得到两个模型分别记为model1.pkl 和model2.pkl.
2.修改network中load model的两个模型路径，然后运行python train1.py得到最终训练模型
ps:第一步训练的网络为Res29_0.py 第二步训练的网络为Res29_1.py，其区别在于有没有SKnet的融合网络 

测试代码:
before use the test code you should modify the following places:
1.on line 75 of test_real.py you should modify the path of model 
2.on line 78 of test_real.py you should modify the path of datas
3.on line 112 of test_real.py you should modify the path of results

if you want to bokeh your images, you can run the test code directly:
python test_real.py



Comments
If you have any questions or comments on my codes, please email to me. 23020191153212@stu.xmu.edu.cn

Reference
[1]. https://arxiv.org/pdf/1903.06586.pdf
