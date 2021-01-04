import json
import jieba
import gensim
import random


def split_word(sentence):
    words = jieba.cut(sentence)
    result = [i for i in words]
    return result


# 文件路径
data_filepath       = '../data/data.json'            # 原始数据
train_filepath      = '../data/data_train.json'      # 训练集
test_filepath       = '../data/data_test.json'       # 测试集
output_filepath     = '../data/output.json'          # 输出

splitdata_filepath  = '../data/splitdata.json'  # 分词结果路径
dictionary_filepath = '../data/dictionary'      # gensim字典路径
model_filepath      = '../data/tfidf.model'     # tfidf模型路径
index_filepath      = '../data/tfidf.index'     # 相似度比较序列路径


# 随机划分数据集
print("> 正在随机划分数据集")
train = []
test = []
with open(data_filepath, encoding='utf-8') as original_file:
    data = json.load(original_file)
    for value in data:
        rand_num = random.uniform(0, 10)
        if rand_num < 8.0:
            train.append(value)
        else:
            test.append(value)

print("  训练集：" + str(len(train)) + "  测试集" + str(len(test)))
with open(train_filepath, 'w', encoding='utf-8') as train_file:
    train_file.write(json.dumps(train, ensure_ascii=False))
with open(test_filepath, 'w', encoding='utf-8') as test_file:
    test_file.write(json.dumps(test, ensure_ascii=False))

with open(train_filepath, encoding='utf-8') as train_file:
    data = json.load(train_file)


# 生成分词结果
print("> 正在分词")
content = []
for value in data:
    question = value['question']
    content.append(split_word(question))
with open(splitdata_filepath, 'w', encoding='utf-8') as f:
    f.write(json.dumps(content, ensure_ascii=False))


# 生成gensim字典
print("> 正在生成gensim字典")
dictionary = gensim.corpora.Dictionary(content)
dictionary.save(dictionary_filepath)
num_features = len(dictionary)  # 特征数


# 生成tfidf模型
print("> 正在生成tfidf模型")
corpus = [dictionary.doc2bow(line) for line in content]  # 语料转词袋表示
tfidf = gensim.models.TfidfModel(corpus)  # 构建tfidf模型
tfidf.save(model_filepath)  # 保存tfidf模型


# 生成tfidf相似度比较序列
print("> 正在生成tfidf相似度比较序列")
index = gensim.similarities.Similarity(index_filepath, tfidf[corpus], num_features)  # 文本相似度序列
index.save(index_filepath)


# 导入测试集进行测试
with open(test_filepath, encoding='utf-8') as test:
    test_data = json.load(test)
print("> 正在使用测试集测试，共有" + str(len(test_data)) + "个问题")

right = 0
output_list = []
for value in test_data:
    question = value['question']
    sentences = split_word(question)     # 分词
    vec = dictionary.doc2bow(sentences)  # 转词袋表示
    sims = index[tfidf[vec]]             # 相似度比较
    sorted_sims = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)  # 逆序
    i = sorted_sims[0][0]

    output_list.append({'question:':value['question'], 'answer':data[i]['answer']}) 
    if(str(data[i]['answer']) == str(value['answer'])):
        right = right + 1

with open(output_filepath, 'w', encoding='utf-8') as output:
    output.write(json.dumps(output_list, ensure_ascii=False))

print("  正确率至少为：" + str(float(right/len(test_data)) * 100) + "%")


# 允许用户继续输入问题
sentences = input('Question: ')
while True:
    sentences = split_word(sentences)    # 分词
    vec = dictionary.doc2bow(sentences)  # 转词袋表示
    sims = index[tfidf[vec]]             # 相似度比较
    sorted_sims = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)  # 逆序
    i = sorted_sims[0][0]
    print(data[i]['answer'])
    sentences = input('Question: ')
