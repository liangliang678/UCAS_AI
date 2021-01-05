# 冬奥会领域问答机器人


本仓库的冬奥会领域问答机器人是中国科学院大学人工智能基础课程的大作业。该问答机器人属于检索式问答系统，采用gensim库和TF-IDF模型+余弦相似度。关于该问答机器人的原理，可以参考doc目录下的实验报告。


## 目录结构
1. ./data  
该目录下是实验使用的数据，train_set.json是训练集，test_set.json是测试集，目录“原始数据”下是老师提供的.xlsx、.csv、.ttl等格式的数据。
2. ./doc  
该目录下是实验报告和汇报使用的实验展示PPT。
3. ./src  
该目录下是问答机器人的源代码。


## 运行方法
1. 确认./data目录下有训练集train_set.json和测试集test_set.json。
2. 在./src下运行main.py，初次运行会在./data下保存多个文件。

| 文件 | 说明 |
| :--- | :--- |
| dictionary | gensim字典 |
| splitdata.json | 分词结果 |
| tfidf.index | 相似度序列 |
| tfidf.index.0 | 相似度序列缓存文件 |
| tfidf.model | TF-IDF模型 |
| output.json | 测试结果输出 |


## 注意事项
1. 使用测试集测试时，终端中输出的“正确数量”和“正确率”只统计输出答案与参考答案完全相同的情况；但是输出答案与参考答案不完全相同的情况也有可能是正确的，可以通过终端中“答案不匹配”的相关输出人工统计正确率。
2. 第二次运行起会直接使用./data下的文件构建模型，如果改变了训练集，请删除上方表格中的文件。


## 参考资料
1. [检索式问答机器人](https://github.com/vba34520/Retrieval-Bot)
2. [【gensim中文教程】开始使用gensim](https://blog.csdn.net/duinodu/article/details/76618638)
3. [TF-IDF算法原理及其使用详解](https://zhuanlan.zhihu.com/p/94446764)


## TODO
1. 得到测试集后测试并完善实验报告
2. 完成PPT
