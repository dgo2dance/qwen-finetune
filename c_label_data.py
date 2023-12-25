import pandas as pd
import json
from multiprocessing import cpu_count, Pool
from tqdm import tqdm
from openai import AzureOpenAI


# Azure实例
client = AzureOpenAI(
    api_version="2023-07-01-preview",
    api_key="d6a7d16408384e04a2769920d3999ef8",
    azure_endpoint="https://qt.openai.azure.com/",
)

# 读取提示词文件
with open("scripts/prompt.txt", "r") as f:
    prompt = f.read()


def get_answer(news, tag, prompt):
    """
    根据新闻与标签，给定提示词，返回GPT的回复
    :param news: str
    :param tag: str
    :param prompt: str
    :return: json
    """
    # 文本拼接
    content = f"news: {news}\ntag: {tag}\n"
    # 消息拼接
    message = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": content},
    ]
    # 获取返回结果
    response = client.chat.completions.create(
        model="gpt35-1106",
        temperature=0,
        response_format={"type": "json_object"},
        messages=message,
    )
    # 转换成Json格式
    result = json.loads(response.choices[0].message.content)
    return result


def process_row(row):
    """
    对每行数据进行标注
    :param row: series
    :return: json
    """
    try:
        return get_answer(row["content"], row["tag_name"], prompt)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        with open("error_log.txt", "a") as log_file:
            log_file.write(f"Error occurred: {str(e)}\n")


def main(sample_num):
    """
    给定新闻数量，进行增量标注
    :param sample_num: int 新标注新闻的数量
    :return:
    """
    # 原始新闻数据文件
    df1 = pd.read_pickle("data/news/merge_df.pkl")
    # 选择字数小于2000的新闻
    df1["len"] = df1["content"].map(len)
    df1 = df1[df1['len'] < 2000]
    # 已有的标注新闻
    df2 = pd.read_pickle('data/news/label_data.pkl')
    # 筛选出未标注的新闻
    remove_index = df2.index
    df = df1.drop(remove_index)
    # 选取特定数量
    chunk = df[["publish_time", "tag_code", "tag_name", "content"]].sample(sample_num)
    # 多进程标注
    n_cores = cpu_count()
    pool = Pool(n_cores)
    results = list(
        tqdm(pool.imap(process_row, chunk.to_dict(orient="records")), total=len(chunk))
    )
    pool.close()
    pool.join()
    # 结果合并
    chunk['result'] = results
    chunk = pd.concat([chunk, pd.json_normalize(chunk['result'])], axis=1)
    chunk = pd.concat([df2, chunk])
    chunk.to_pickle(f'data/label_data.pkl')


if __name__ == "__main__":
    main(2000)
