import sys
import random

"""
将data目录下的qestion和answer文件按8:2的比例分为训练集和测试集
"""
with open('../data/question', 'r', encoding='utf-8') as question_file:
    with open('../data/answer', 'r', encoding='utf-8') as answer_file:
        with open('../data/train_set_question', 'w', encoding='utf-8') as train_question_file:
            with open('../data/train_set_answer', 'w', encoding='utf-8') as train_answer_file:
                with open('../data/test_set_question', 'w', encoding='utf-8') as test_question_file:
                    with open('../data/test_set_answer', 'w', encoding='utf-8') as test_answer_file:
                        while True:
                            question = question_file.readline()
                            answer = answer_file.readline()
                            if question and answer:
                                rand_num = random.uniform(0, 10)
                                if rand_num < 8.0:
                                    train_question_file.write(question)
                                    train_answer_file.write(answer)
                                else:
                                    test_question_file.write(question)
                                    test_answer_file.write(answer)
                            else:
                                break