# 文件结构


# 文件说明
bezier_curve.py定义了具体的bezier曲线，分段插值方法以及障碍物碰撞判断等函数

evalution.py为评估模块，包括欧式距离，曲线曲率等维度的评估，生成图片保存在image中以可视化

image_geneate.py 为标准输入图片生成模块，负责生成黑白顶点网络图，并保存在image下作为后续程序输入

main.py 用于用各模块生成路径规划结构

plot.py是一些绘图模块

view_hev.py用于查看一张图片的hsv

image用于保存各模块输入和输出图片



# 操作说明
运行image_generate.py在iamge文夹下生成一张输入图片，名为mazex.png，其中变量x依次排列

运行main.py并输入对应image文件夹下待生成路径图片mazex.png，并在弹窗中依次点击起点，终点，会在image文件夹下生成gen_mazex.png以及cur_mazex.png
,前者为左右布局图，左侧为用A  star算法生成的初始折线和膨胀障碍物后用相同算法生成的折线 ，后者为在膨胀障碍物生成折线的基础上用bezier细分方法插值得到的bezier曲线
后者为左右布局图，左为gen_mazex.png右侧图片，右侧为左侧图片曲线在各处的曲率变化

需要主要的是,运行main.py，重复输入同名mazex.png，会覆盖同名的原有gen_mazex.png和cur_mazex.png。



# 版本