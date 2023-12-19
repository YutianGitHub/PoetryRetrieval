import json
import os
from typing import List, Dict, AnyStr, Any, Tuple, Iterable
import itertools
import numpy as np
import pandas as pd
import tqdm
import zhipuai
from matplotlib import pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import ast
from numpy.linalg import norm
zhipuai.api_key = "7cdeb7f014f39871caac51994948707d.C3tQB414seOfHgfC"

def get_all_files_in_directory(directory: str, rules=lambda x: True) -> List[AnyStr]:
    """
    返回一个文件夹的所有文件路径
    :param directory: 文件夹路径
    :param rules: 过滤文件名规则
    :return: List[AnyStr]
    """
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if rules(file):
                file_list.append(os.path.join(root, file))
    return file_list


def parse_json(json_path: str) -> dict:
    try:
        with open(json_path, "r", encoding="utf-8", errors="ignore") as f:
            return json.load(f)
    except Exception as e:
        print(f"解析文件 {json_path} 时出错：{e}")
        return None


def json_list_parse(it: Iterable, multi_process=False, num_workers=4) -> List[Dict]:
    """
    一系列文件路径，用json解析，返回一个Dict列表
    :param it: 文件路径列表
    :return:
    """
    if not multi_process:
        print("单线程解析...")
        res = list()
        for fp in tqdm.tqdm(it, total=len(it)):
            res.append(parse_json(fp))
    else:
        print(f"{num_workers}线程解析")
        res = list()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(parse_json, path) for path in it]
            for future in tqdm.tqdm(futures, total=len(it)):
                result = future.result()
                if result:
                    res.append(result)

    return res


def to_csv(df: pd.DataFrame, file_name="your.csv", **kwargs) -> None:
    df.to_csv(file_name, index=False, **kwargs)


def read_csv(file_name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(file_name, **kwargs)


def merge_list(ls: List[List[Any]]) -> List[Any]:
    """
    铺平二级列表
    :param ls:二级列表
    :return: 平列表
    """

    assert len(ls) >= 1

    base = ls[0]
    if type(base) != "list":
        return ls
    for i in range(1, len(ls)):
        base.extend(ls[i])
    return base


def chunks(iterable, batch_size=100):
    """A helper function to break an iterable into chunks of size batch_size.
    :param batch_size: 每chunk的长度
    :param iterable:可迭代的对象
    :return: 生成器
    """
    it = iter(iterable)
    chunk = list(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = list(itertools.islice(it, batch_size))


def len_row(row: pd.Series) -> int:
    """
    一行dataframe的字符长度
    :param row:
    :return:
    """
    s = 0
    for k in row.keys():
        s += len_object(row[k])
    return s


def series_hist(ser: pd.Series, filter_len=2, **kwargs) -> pd.Series:
    """
    绘画字符串长度分布直方图
    :param ser: Series对象
    :param filter_len:过滤长度
    :param kwargs:
    :return:
    """
    lengths = ser.str.len()
    filtered = lengths.loc[~(lengths <= filter_len)]
    plt.hist(filtered, edgecolor='black', **kwargs)
    plt.xlabel('String Length')
    plt.ylabel('Frequency')
    plt.title('String Length Histogram')
    plt.show()
    return filtered


def str2dict(s: AnyStr) -> Dict:
    """
    str转换dict
    :param s:
    :return:
    """
    if isinstance(s, Dict):
        return s
    return ast.literal_eval(s)


def len_object(d: Any) -> int:
    s = 0
    if isinstance(d, int):
        return 1
    if not isinstance(d, dict):
        return len(str(d))
    for v in d.values():
        s += len_object(v)
    return s


def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def dot_product(a, b):
    return np.dot(a, b)


async def asyncChatChatGLM(message, history, temp=0.95, top_p=0.7, tag=0)->Tuple:
    history_glm_format = []
    for human, assistant in history:
        history_glm_format.append({"role": "user", "content": human})
        history_glm_format.append({"role": "assistant", "content": assistant})
    history_glm_format.append({"role": "user", "content": message})

    responses = zhipuai.model_api.sse_invoke(
        model="chatglm_turbo",
        prompt=history_glm_format,
        temperature=temp,
        top_p=top_p,
        incremental=True
    )

    buffer = ""
    for event in responses.events():
        buffer += event.data
        if event.event == "finish":
            print(f"tag:{tag} finish.")
    return tag, buffer


def syncChatChatGLM(message, history, temp=0.95, top_p=0.7, tag=0)->Tuple:
    history_glm_format = []
    for human, assistant in history:
        history_glm_format.append({"role": "user", "content": human})
        history_glm_format.append({"role": "assistant", "content": assistant})
    history_glm_format.append({"role": "user", "content": message})

    responses = zhipuai.model_api.sse_invoke(
        model="chatglm_turbo",
        prompt=history_glm_format,
        temperature=temp,
        top_p=top_p,
        incremental=True
    )

    buffer = ""
    for event in responses.events():
        buffer += event.data
        if event.event == "finish":
            print(f"tag:{tag} finish.")
    return tag, buffer