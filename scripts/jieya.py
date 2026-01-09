import itertools
import json
import os
import tarfile
from io import StringIO, BytesIO
from parser import ParserError
from typing import Dict, Optional, List

import gzip
import numpy as np
import pandas as pd
from sklearn import logger


def get_best_res(df, file_name):
    df_cleaned = df.dropna(subset=['model_name', 'file_name', 'strategy_args', 'mse_norm'])

    def extract_horizon(strategy_args):
        try:
            return json.loads(strategy_args).get('horizon', None)
        except (ValueError, TypeError):
            return None

    df_cleaned['horizon'] = df_cleaned['strategy_args'].apply(extract_horizon)
    df_cleaned = df_cleaned.dropna(subset=['horizon'])

    # 按分组排序并取每组前5条
    best_results = (
        df_cleaned
        .sort_values(['model_name', 'file_name', 'horizon', 'mse_norm'], ascending=[True, True, True, True])
        .groupby(['model_name', 'file_name', 'horizon'])
        .head(1)
        .reset_index(drop=True)[['model_name', 'strategy_args', 'model_params', 'mse_norm', 'mae_norm', 'file_name']]
    )

    dir_name = os.path.dirname(file_name)
    base_name = os.path.basename(file_name)
    new_file_path = os.path.join(dir_name, base_name)
    best_results.to_csv(new_file_path, index=False)

    return new_file_path

class FieldNames:

    MODEL_NAME = "model_name"
    FILE_NAME = "file_name"
    MODEL_PARAMS = "model_params"
    STRATEGY_ARGS = "strategy_args"
    FIT_TIME = "fit_time"
    INFERENCE_TIME = "inference_time"
    ACTUAL_DATA = "actual_data"
    INFERENCE_DATA = "inference_data"
    LOG_INFO = "log_info"

    @classmethod
    def all_fields(cls) -> List[str]:
        return [
            cls.MODEL_NAME,
            cls.FILE_NAME,
            cls.MODEL_PARAMS,
            cls.STRATEGY_ARGS,
            cls.FIT_TIME,
            cls.INFERENCE_TIME,
            cls.ACTUAL_DATA,
            cls.INFERENCE_DATA,
            cls.LOG_INFO,
        ]

def _find_log_files(directory: str) -> List[str]:
    log_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            # TODO: this is a temporary solution, any good methods to identify a log file?
            if file.endswith(".csv") or file.endswith(".tar.gz"):
                log_files.append(os.path.join(root, file))
    return log_files

ARTIFACT_COLUMNS = [
    # FieldNames.ACTUAL_DATA,
    # FieldNames.INFERENCE_DATA,
    # FieldNames.LOG_INFO,
]

ARTIFACT_COLUMNS1 = [
    FieldNames.LOG_INFO,
]

def _load_log_data(log_files: List[str]) -> pd.DataFrame:
    log_files = itertools.chain.from_iterable(
        [[fn] if not os.path.isdir(fn) else _find_log_files(fn) for fn in log_files]
    )

    ret = []
    for fn in log_files:
        logger.info("loading log file %s", fn)
        try:
            res = read_log_file(fn).drop(columns=ARTIFACT_COLUMNS)
            # b = res['log_info'].values[0]
            # if type(b) == str:
            #     print(f'{fn} has errors')
            #     saved_str = f'{fn} has errors\n'
            #     save_log_path = r"/Users/xiangfeiqiu/D/OTB_final_results/newest/log.txt"
            #     with open(save_log_path, 'a') as file:
            #         file.write(saved_str)
            #     continue
            # else:
            #     res = res.drop(columns=ARTIFACT_COLUMNS1)
            ret.append(res)
        except (FileNotFoundError, PermissionError, KeyError, ParserError):
            # TODO: it is ugly to identify log files by artifact columns...
            logger.info("unrecognized log file format, skipping %s...", fn)
    return pd.concat(ret, axis=0)
def read_log_file(fn: str) -> pd.DataFrame:
    ext = os.path.splitext(fn)[1]
    compress_method = get_compress_method_from_ext(ext)
    # compress_method = 'gz'

    if compress_method is None:
        return pd.read_csv(fn)
    else:
        with open(fn, "rb") as fh:
            data = fh.read()
        data = decompress(data, method=compress_method)
        ret = []
        for k, v in data.items():
            ret.append(pd.read_csv(StringIO(v)))
        return pd.concat(ret, axis=0)



def compress_gz(data: Dict[str, str]) -> bytes:
    """
    Compress in gz format
    """
    outbuf = BytesIO()

    with tarfile.open(fileobj=outbuf, mode="w:gz") as tar:
        for k, v in data.items():
            info = tarfile.TarInfo(name=k)
            v_bytes = v.encode("utf8")
            info.size = len(v_bytes)
            tar.addfile(info, fileobj=BytesIO(v_bytes))

    return outbuf.getvalue()

def compress_gzip(data: Dict[str, str]) -> bytes:
    """
    Compress data using Gzip compression.
    """
    outbuf = BytesIO()

    with gzip.GzipFile(fileobj=outbuf, mode="wb") as gz:
        for k, v in data.items():
            v_bytes = v.encode("utf8")
            gz.write(v_bytes)

    return outbuf.getvalue()


def decompress_gzip(compressed_data: bytes) -> Dict[str, str]:
    """
    Decompress Gzip-compressed data and return the original dictionary.
    """
    decompressed_data = {}
    compressed_buf = BytesIO(compressed_data)

    with gzip.GzipFile(fileobj=compressed_buf, mode="rb") as gz:
        while True:
            chunk = gz.read(1024)  # Read a chunk of decompressed data (adjust chunk size if needed)
            if not chunk:
                break  # No more data to read
            chunk_str = chunk.decode("utf8")
            key_values = chunk_str.split("\n")

            for key_value in key_values:
                if key_value:
                    key, value = key_value.split(":")
                    decompressed_data[key] = value

    return decompressed_data


def decompress_gz(data: bytes) -> Dict[str, str]:
    ret = {}
    with tarfile.open(fileobj=BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile():
                ret[member.name] = tar.extractfile(member).read().decode("utf8")

    return ret


def compress(data: Dict[str, str], method: str = "gz") -> bytes:
    if method != "gz":
        compress_gzip(data)
        # raise NotImplementedError("Only 'gz' method is supported by now")
    return compress_gz(data)


def decompress(data: bytes, method: str = "gz") -> Dict[str, str]:
    if method != "gz":
        # decompress_gzip(data)
        raise NotImplementedError("Only 'gz' method is supported by now")
    return decompress_gz(data)


def get_compress_file_ext(method: str) -> str:
    if method != "gz":
        return "gzip"
        # raise NotImplementedError("Only 'gz' method is supported by now")
    return "tar.gz"


def get_compress_method_from_ext(ext: str) -> Optional[str]:
    return {
        ".gz": "gz"
    }.get(ext)

def convert_gz_to_csv(folder_path):


    compressed_files_path = []

    # 遍历文件夹及其子文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 构建文件路径
            file_path = os.path.join(root, file)
            # 添加条件判断：文件的开头不为“._”并且结尾为“.tar”
            if not file.startswith("._") and file.endswith(".gz"):
                # 将满足条件的文件路径添加到列表
                compressed_files_path.append(file_path)


    compressed_files_path1 = []
    for file_path in compressed_files_path:
        new_list = []
        new_list.append(file_path)
        try:
            a = _load_log_data(new_list)
            compressed_files_path1.append(file_path)
        except:
            print('******************************************************')
            print(file_path)
            print('******************************************************')


    print(len(compressed_files_path1))
    a = _load_log_data(compressed_files_path1)
    return a



log_data = convert_gz_to_csv(r'/home/SEER-ste/result')
log_data.to_csv('/home/SEER-ste/result/result.csv', index=False)
# get_best_res(log_data, "/home/TFB/result/sparsetsf-hdmixer-best.csv")
