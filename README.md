# 冬奥会领域问答机器人

## 作业要求
2021年1月10日：提交代码和文档报告   
2021年1月12日：现场使用PPT介绍方案、运行演示、展示效果  

## 参考资料
[中文聊天机器人的实现](https://blog.csdn.net/zzZ_CMing/article/details/81316033)

[python chatterbot库](https://github.com/gunthercox/ChatterBot)

[python chatterbot库使用教程](https://blog.csdn.net/LHWorldBlog/article/details/81039399)

## TODO
1. 第一个参考资料的聊天机器人看起来比较简单一点。数据集已经转换为该项目要求的格式，在./data目录下。目前我们需要按照第一个参考资料进行模型训练和模型测试。
2. 如果第一个参考资料的聊天机器人可行，我们需要改变其输入输出格式。作业要求的输入输出格式为json。
3. 按照实验要求，我们需要将给定的问答对随机划分为数据集、验证集和测试集（比例为7：1：2或8：1：1），随机划分的模块好像需要自己写。但是第一个参考资料的聊天机器人不需要验证集，直接划分数据集：测试集为8：2也行？[关于验证集与测试集的区别](https://www.cnblogs.com/morwing/p/12144476.html)。