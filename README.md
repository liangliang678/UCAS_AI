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
加载file_list中的所有语句，使用[jieba库](https://github.com/fxsjy/jieba)切分成词语后统计词频，按词频由高到低排序后顺次编号并存到word2id_dict和id2word_dict中。由此就建立了词库，即word和id之间的双向映射。注意，超过最小词频min_freq的词才会存入列表。
2. word2id(word)  
由word返回id，若列表中没有该word则返回none
3. id2word(id)  
由id返回word

我们在全局中使用load_file_list将源文件处理好并得到max_token_id。

### main.py
#### get_id_list_from(sentence)
使用jieba库将sentence进行分词，将得到的每一个word的id加入一个列表中，返回该列表。

#### get_train_set()
将每一个问答对转换为id列表，由[question_id_list, answer_id_list]的形式存储在列表train_set中，返回train_set。注意，answer_id_list的末尾添加了一个EOS_ID表示结束。

#### get_samples(train_set, batch_num)
将train_set中的batch_num个问答对转化为向量的形式。矩阵encoder_input的行向量是数组，每一个列向量是问题；矩阵decoder_input的行向量是数组，每一个列向量是回答；矩阵target_weight的每一项表示矩阵decoder_input的每一项是否有意义。