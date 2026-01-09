import json
import os

import pandas as pd

horizon_kind_num = 4
top_n = 5
folder_path = 'data'

# ====================================================================

save_name = f"best_params_new_data_top{top_n}.csv"
csv_files = [os.path.join(folder_path, f)
             for f in os.listdir(folder_path)
             if f.endswith('.csv')]
df_list = [pd.read_csv(file) for file in csv_files]
result = pd.concat(df_list, ignore_index=True)
result['strategy_args_dict'] = result['strategy_args'].apply(json.loads)

# # 提取 target_channel 列
# result['target_channel'] = result['strategy_args_dict'].apply(lambda x: x.get('target_channel'))
#
# # 筛选 target_channel 不为 None 或 空字符串的行
# result = result[result['target_channel'].notnull() & (result['target_channel'] != '')]

idx = result.groupby(
    ['model_name', 'strategy_args', 'model_params', 'file_name']
)['mse_norm'].idxmin()

result = result.loc[idx].reset_index(drop=True)

result["model_params"] = result["model_params"].apply(lambda x: x.replace(" ", ""))


def get_horizon(s):
    d = json.loads(s)
    return d.get('horizon', None)


result['horizon'] = result['model_params'].apply(get_horizon)


def process_strategy_args(s):
    d = json.loads(s)
    d.pop('horizon', None)
    d.pop('seq_len', None)
    return str(d)


result['clean_model_params'] = result['model_params'].apply(process_strategy_args)

grouped = result.groupby(['model_name', 'clean_model_params', 'file_name'])['strategy_args'].nunique().reset_index()
valid_groups = grouped[grouped['strategy_args'] == horizon_kind_num]

# 将这些组合用于筛选原始数据
result = result.merge(valid_groups, on=['model_name', 'clean_model_params', 'file_name'], how='inner')

mean_mse = (
    result.groupby(['file_name', 'clean_model_params', 'model_name'])['mse_norm']
    .mean()
    .reset_index(name='mean_mse_norm')
)

# 2. 对每个 file_name 和 model_name，取平均 mse_norm 最低的前三个 model_params
top_params = (
    mean_mse
    .groupby(['file_name', 'model_name'])
    .apply(lambda g: g.nsmallest(top_n, 'mean_mse_norm'))
    .reset_index(drop=True)
)

# 3. 用筛选出的 file_name, model_name, model_params 去原始数据中筛选对应的行
result = result.merge(
    top_params[['file_name', 'model_name', 'clean_model_params', 'mean_mse_norm']],
    on=['file_name', 'model_name', 'clean_model_params'],
    how='inner'
)

result = result[["model_name", "strategy_args_x", "model_params", "clean_model_params",
                 "mse_norm", "mae_norm", "file_name", "horizon", 'mean_mse_norm']]

result = result.sort_values(by=["model_name", "file_name", 'mean_mse_norm', "clean_model_params", "horizon"])
result.to_csv(save_name, index=False)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
