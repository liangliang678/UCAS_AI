# 冬奥会领域问答机器人

## 作业要求
2021年1月10日：提交代码和文档报告   
2021年1月12日：现场使用PPT介绍方案、运行演示、展示效果  

## 参考资料
[中文聊天机器人的实现](https://blog.csdn.net/zzZ_CMing/article/details/81316033)

## TODO
1. 生成模型
2. 改变源代码的输出格式
3. 分析代码

## 代码分析
### separate_data.py
该程序将data目录下的qestion和answer按8:2的比例分为训练集和测试集，训练集的问题和回答为train_set_question和train_set_anwser，测试集的问题和参考回答为test_set_question和test_set_anwser。

### word_token.py
WordToken类有两个列表word2id_dict、id2word_dict，三种方法：
1. load_file_list(file_list, min_freq)  
加载file_list中的所有句子，使用[jieba库](https://github.com/fxsjy/jieba)切分成词语。统计每个词的词频，按词频由高到低排序后顺次编号并存到word2id_dict和id2word_dict中。由此就建立了词库，即word和id之间的双向映射。注意，超过最小词频min_freq的词才会存入列表。
2. word2id(word)  
由word返回id，若列表中没有该word则返回None
3. id2word(id)  
由id返回word，若列表中没有该id则返回None

### main.py
在全局位置调用wordToken.load_file_list建立词库并计算出num_encoder_symbols，num_decoder_symbols。
参数为train时调用train()训练，参数为test时调用test()测试。

### 训练过程
首先通过get_train_set()得到训练问答集。

### 测试过程
