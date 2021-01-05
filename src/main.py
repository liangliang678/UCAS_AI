import json
import jieba
import gensim


def split_word(sentence):
    words = jieba.cut(sentence)
    result = [word for word in words]
    return result


# 文件路径
train_filepath      = '../data/train_set.json'      # 训练集路径
test_filepath       = '../data/test_set.json'       # 测试集路径
output_filepath     = '../data/output.json'         # 输出路径

splitdata_filepath  = '../data/splitdata.json'  # 分词结果路径
dictionary_filepath = '../data/dictionary'      # gensim字典路径
model_filepath      = '../data/tfidf.model'     # tfidf模型路径
index_filepath      = '../data/tfidf.index'     # 相似度比较序列路径


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
num_features = len(dictionary)
corpus = [dictionary.doc2bow(line) for line in content]


# 生成tfidf模型
print("> 正在生成tfidf模型")
tfidf = gensim.models.TfidfModel(corpus)
tfidf.save(model_filepath)


# 生成tfidf相似度比较序列
print("> 正在生成tfidf相似度比较序列")
index = gensim.similarities.Similarity(index_filepath, tfidf[corpus], num_features)
index.save(index_filepath)


# 导入测试集进行测试
with open(test_filepath, encoding='utf-8') as test_file:
    test_data = json.load(test_file)
print("> 正在使用测试集测试，共有" + str(len(test_data)) + "个问题")

right = 0
output_list = []
for value in test_data:
    question = value['question']
    sentences = split_word(question)     # 分词
    vec = dictionary.doc2bow(sentences)  # 转词袋表示
    sims = index[tfidf[vec]]             # 相似度比较
    sorted_sims = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
    i = sorted_sims[0][0]

    output_list.append({'question:':value['question'], 'answer':data[i]['answer']}) 
    if(str(data[i]['answer']) == str(value['answer'])):
        right = right + 1
    else:
        print("  答案不匹配：输出\""+str(data[i]['answer'])+"\"")
        print("              参考\"" + str(value['answer']) + "\"")

with open(output_filepath, 'w', encoding='utf-8') as output:
    output.write(json.dumps(output_list, ensure_ascii=False))

print("  正确数量" + str(right) + " / " + str(len(test_data)))


# 允许用户继续输入问题
sentences = input('Question: ')
while True:
    sentences = split_word(sentences)    # 分词
    vec = dictionary.doc2bow(sentences)  # 转词袋表示
    sims = index[tfidf[vec]]             # 相似度比较
    sorted_sims = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
    i = sorted_sims[0][0]
    print(data[i]['answer'])
    sentences = input('Question: ')
