Example-based-Bokeh-Effect-Challenge

Requirement
Python 3.7.3
torch 0.4.1
scipy 1.3.1
numpy 1.16.4
imageio 2.5.0
cv 2 4.1.0

dataset download
train dataset��https://pan.baidu.com/s/1iB4N93L_SFRAoEHEm4pIOQ
test  dataset��https://pan.baidu.com/s/1POpip1m0JDiQU4wodoCnMg

ѵ������:
1.�������޸�·����Ȼ����dataset/data.py���޸�ͼƬ��С����һ���޸�Ϊresize(256*256),
��һ���޸�Ϊcrop(512*512)�ֱ�����python train0.py�õ�����ģ�ͷֱ��Ϊmodel1.pkl ��model2.pkl.
2.�޸�network��load model������ģ��·����Ȼ������python train1.py�õ�����ѵ��ģ��
ps:��һ��ѵ��������ΪRes29_0.py �ڶ���ѵ��������ΪRes29_1.py��������������û��SKnet���ں����� 

���Դ���:
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
