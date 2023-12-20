import importlib
import time
import re
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
poets1_1 = []
poets2_1 = []

pattern = re.compile(r'诗人: ([\u4e00-\u9fa5]+)\n')

spliter = RecursiveCharacterTextSplitter(
    chunk_size = 512,
    chunk_overlap  = 64,
    length_function = len,
    is_separator_regex = True,
    separators=["\n\n","\n"," ",""]
)

for p in tqdm.tqdm(poets2,total=len(poets2)):

    match = pattern.search(p.page_content)
    if match:
        name = match.group(1)
    else:
        print(f"\nnot find name:{p.page_content[0:20]}")
        continue

    head = f"诗人：{name}的生平片段"
    if len(p.page_content)<=500:
        poets2_1.append(head+p.page_content)
    else:
        texts = spliter.split_text(p.page_content)
        for i,text in enumerate(texts):
            poets2_1.append(head+f"{i+1}：\n"+text)

print(poets2_1[1])


shortObj = []
iterObj = poets1



concur = 3
res = dict()
count = 0
chunks = util.chunks(iterObj, batch_size=concur)

for batch in tqdm.tqdm(chunks, total=len(iterObj) // concur):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 调用你的函数，并传入不同的参数
        # 这里的参数可以是不同的消息、历史记录等
        # 返回的futures列表包含每个调用的结果
        futures = [executor.submit(util.syncChatChatGLM,
                                   prompt.format(content=batch[tag].page_content),
                                   [], 0.95, 0.7, count + tag) for tag in range(concur)]
        for i in range(concur):
            res[count + i] = {"文档": batch[i].page_content}
        count += concur
        print()
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                # 获取函数的返回值

                # 关键问题：返回顺序不一定一致！

                result = future.result()
                res[result[0]]["摘要"] = result[1]
                print(f"Tag:{result[0]},Result: {result}")
            except Exception as e:
                print(f"Error: {e}")
        time.sleep(1.2)