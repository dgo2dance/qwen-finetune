import pandas as pd
import os
import numpy as np
import json
import shutil


def process_data(data_path, train_path, train_num, test_path, test_num):
    """

    :param data_path:
    :param train_path:
    :param train_num:
    :param test_path:
    :param test_num:
    :return:
    """
    chunk_num = train_num + 1

    # 预处理数据文件夹
    def process_folder(path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    process_folder(train_path)
    process_folder(test_path)
    # 数据随机重排
    data = pd.read_pickle(data_path).sample(frac=1).reset_index(drop=True)
    # 每个训练集的大小
    chunk_size = len(data) // chunk_num
    # 数据分割的索引点
    chunk_split_points = [i * chunk_size for i in range(1, chunk_num)]
    # 分割df的列表
    df_split = np.split(data, chunk_split_points)

    # 生成测试集
    test_data = df_split[0]

    # 格式化
    def get_question(row):
        question = f"[[{row['tag_name']}]]: {row['content']}"
        return question

    test_data['question'] = test_data.apply(get_question, axis=1)
    # 将result, reason, sentiment, impact列添加_old后缀
    rename_dict = {col: col + '_old' for col in ['result', 'reason', 'sentiment', 'impact']}
    test_data = test_data.rename(columns=rename_dict)
    # 测试集分割
    test_size = len(test_data) // test_num
    test_split_points = [i * test_size for i in range(1, test_num)]
    test_split = np.split(test_data, test_split_points)
    for i, test in enumerate(test_split):
        test.to_pickle(os.path.join(test_path, f'test-{i:02d}.pkl'))

    # 生成训练集
    for i, df in enumerate(df_split[1:]):
        dict_list = []
        for ind, row in df.iterrows():
            element_dict = {'id': str(ind).zfill(4)}
            prompt = f"[[{row['tag_name']}]]: {row['content']}"
            conversation_dict1 = {"from": "user", "value": prompt}
            conversation_dict2 = {"from": "assistant", "value": str(row['result'])}
            conversation_list = [conversation_dict1, conversation_dict2]
            element_dict['conversations'] = conversation_list
            dict_list.append(element_dict)
        # 保存为json文件
        file_path = os.path.join(train_path, f"train_{str(i).zfill(2)}.json")
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(dict_list, json_file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    data_path = 'data/news/label_data.pkl'
    train_path = 'data/train'
    test_path = 'data/test'
    train_num = 5
    test_num = 10
    process_data(data_path, train_path, train_num, test_path, test_num)
