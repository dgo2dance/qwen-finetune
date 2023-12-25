import pandas as pd
import re


news_columns = ['item_id', 'content', 'publish_time']
news_df = pd.read_pickle('data/news.pkl')[news_columns]
tag_columns = ['news_id', 'level1_code', 'tag_code', 'tag_name', 'emotion', 'category']
tags_df = pd.read_pickle('data/tags.pck')[tag_columns]
tags_df = tags_df[tags_df['category'] == 'STOCK'].drop('category', axis=1)


def remove_html_tags(text):
    clean = re.compile(r'(<.*?>|\n|\r|&nbsp;|@|,|")')
    return re.sub(clean, '', text)


merge_df = pd.merge(news_df, tags_df, 'inner', left_on='item_id', right_on='news_id').drop('news_id', axis=1)
merge_df['content'] = merge_df['content'].map(remove_html_tags)
merge_df.to_pickle('data/merge_df.pickle')
