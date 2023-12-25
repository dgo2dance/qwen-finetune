import pandas as pd
import json
from multiprocessing import cpu_count, Pool
from tqdm import tqdm
from openai import AzureOpenAI


client = AzureOpenAI(
    api_version="2023-07-01-preview",
    api_key="d6a7d16408384e04a2769920d3999ef8",
    azure_endpoint="https://qt.openai.azure.com/",
)

with open("scripts/prompt.txt", "r") as f:
    prompt = f.read()


def get_answer(news, tag, prompt):
    content = f"news: {news}\ntag: {tag}\n"
    message = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": content},
    ]
    response = client.chat.completions.create(
        model="gpt35-1106",
        temperature=0,
        response_format={"type": "json_object"},
        messages=message,
    )
    result = json.loads(response.choices[0].message.content)
    return result


def process_row(row):
    try:
        return get_answer(row["content"], row["tag_name"], prompt)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        with open("error_log.txt", "a") as log_file:
            log_file.write(f"Error occurred: {str(e)}\n")


def main(sample_num):
    df1 = pd.read_pickle("data/news/merge_df.pkl")
    df1["len"] = df1["content"].map(len)
    df1 = df1[df1['len'] < 2000]
    df2 = pd.read_pickle('data/news/label_data.pkl')
    remove_index = df2.index
    df = df1.drop(remove_index)
    chunk = df[["publish_time", "tag_code", "tag_name", "content"]].sample(sample_num)
    n_cores = cpu_count()
    pool = Pool(n_cores)
    results = list(
        tqdm(pool.imap(process_row, chunk.to_dict(orient="records")), total=len(chunk))
    )
    pool.close()
    pool.join()
    chunk['result'] = results
    chunk = pd.concat([chunk, pd.json_normalize(chunk['result'])], axis=1)
    chunk = pd.concat([df2, chunk])
    chunk.to_pickle(f'data/label_data.pkl')


if __name__ == "__main__":
    main(2000)
