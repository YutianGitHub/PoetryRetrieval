import asyncio
import importlib
import time

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings

import pandas as pd
import tqdm
import util
import uuid

from langchain.text_splitter import RecursiveCharacterTextSplitter

import concurrent.futures

prompt = """
你是一位资深的文字专家，请为以下内容提供用于检索的标题和关键词：

字数要求： 共50字左右 

标题要求： 突出原文的主题和要点，使用简洁明了的语言表达。能概括文档的各个内容，方便用于检索。 关键词突出原文的要点。如果原文没有明显的信息，你的回复是：无。
输出样例：
【标题】：你生成的标题。
【关键词】：你生成的关键词。

以下是原文：
{content}

"""

poets1 = CSVLoader(file_path="4个数据集/poets_3154_copy.csv", encoding="utf-8").load()
poets2 = CSVLoader(file_path="4个数据集/poets_contents_3154.csv", encoding="utf-8").load()

poets1_long = [p for p in poets1 if p.page_content.__len__() >= 128]
poets2_long = [p for p in poets2 if p.page_content.__len__() >= 128]
chunks = util.chunks(poets1_long, batch_size=3)
res = []
count = 1
for batch in tqdm.tqdm(chunks, total=len(poets1_long) // 3):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 调用你的函数，并传入不同的参数
        # 这里的参数可以是不同的消息、历史记录等
        # 返回的futures列表包含每个调用的结果
        futures = [executor.submit(util.syncChatChatGLM,
                                   prompt.format(content=batch[tag].page_content),
                                   [], 0.95, 0.7, count+tag) for tag in range(3)]
        count += 3
        print()
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                # 获取函数的返回值
                result = future.result()
                res.append(result[1] + "\n\n" + batch[i].page_content)
                print(f"Count:{count},Result: {result}")
            except Exception as e:
                print(f"Error: {e}")
        time.sleep(1)
