import ast
import json
import os

import pandas as pd

MODEL_NAME_MAP = {
    "Crossformer": "time_series_library.Crossformer",
    "DLinear": "time_series_library.DLinear",
    "FEDformer": "time_series_library.FEDformer",
    "FiLM": "time_series_library.FiLM",
    "Informer": "time_series_library.Informer",
    "iTransformer": "time_series_library.iTransformer",
    "MICN": "time_series_library.MICN",
    "NLinear": "time_series_library.NLinear",
    "Nonstationary_Transformer": "time_series_library.Nonstationary_Transformer",
    "PatchTST": "time_series_library.PatchTST",
    "TimeMixer": "time_series_library.TimeMixer",
    "TimesNet": "time_series_library.TimesNet",
    "Triformer": "time_series_library.Triformer",

    "RegressionModel": "darts.RegressionModel",
    "RNNModel": "darts.RNNModel",
    "TCNModel": "darts.TCNModel",

    "VAR_model": "self_impl.VAR_model",

    "FITS": "fits.FITS",
    "PDF": "PDF.PDF",
    "Pathformer": "pathformer.Pathformer",
    "DUET": "duet.DUET",
    "ADARNN": "adarnn.ADARNN",
    "APFormer": "apformer.APFormer",
    "COV": "cov.COV",
    "CrossLinear": "time_series_library.CrossLinear",
    "Amplifier": "amplifier.Amplifier",
    "TimeKAN": "timekan.TimeKAN",
    "TimeXer": "timexer.TimeXer",
    "TiDE1": "time_series_library.TiDE1",
    "xPatch": "xpatch.xPatch",
    "TemporalFusionTransformer1": "time_series_library.TemporalFusionTransformer1",

    "GCGNet_w_mlphead": "GCGNet.GCGNet",
    "GCGNet": "GCGNet.GCGNet",
    "SEER": "seer_modify.SEER",
    "SparseTSF": "sparsetsf.SparseTSF",
    "HDMixer": "hdmixer.HDMixer"
}

MODEL_ADAPTER = {
    "Crossformer": "transformer_adapter",
    "DLinear": "transformer_adapter",
    "FEDformer": "transformer_adapter",
    "FiLM": "transformer_adapter",
    "Informer": "transformer_adapter",
    "iTransformer": "transformer_adapter",
    "MICN": "transformer_adapter",
    "NLinear": "transformer_adapter",
    "Nonstationary_Transformer": "transformer_adapter",
    "PatchTST": "transformer_adapter",
    "TimeMixer": "transformer_adapter",
    "TimesNet": "transformer_adapter",
    "Triformer": "transformer_adapter",
    "CrossLinear": "transformer_adapter",
    "TiDE1": "transformer_adapter",
    "TemporalFusionTransformer1": "transformer_adapter",
}


def generate_script(row):
    model_name = row['model_name']
    model_module_name = MODEL_NAME_MAP[model_name]
    model_params = row['model_params']
    strategy_args = ast.literal_eval(row["strategy_args"].replace('false', 'False').replace('null', 'None'))
    horizon = strategy_args["horizon"]
    # target_channel = strategy_args["target_channel"]
    # strategy_args = json.dumps({"horizon": horizon, "target_channel": target_channel})
    strategy_args = json.dumps({"horizon": horizon})

    file_name = row["file_name"]
    dataset_name = os.path.splitext(file_name)[0]
    adapter = MODEL_ADAPTER.get(model_name, None)
    adapter_part = f' --adapter "{adapter}"' if adapter is not None else ""

    script = (
        f'python ./scripts/run_benchmark.py '
        f'--config-path "rolling_forecast_config.json" '
        f'--data-name-list "{file_name}" '
        f'--strategy-args \'{strategy_args}\'{adapter_part} '
        f'--model-name "{model_module_name}" '
        f'--model-hyper-params \'{model_params}\' '
        f'--gpus 0 --num-workers 1 --timeout 60000 '
        f'--save-path "{dataset_name}/{model_name}"'
    )

    return script


def result2sh(file_name):
    name = os.path.splitext(file_name)[0]
    result = pd.read_csv(file_name)

    for index, row in result.iterrows():
        script = generate_script(row)
        model_name = row["model_name"]
        file_name = row["file_name"]
        dataset_name = os.path.splitext(file_name)[0]
        path = f'{name}_script/{dataset_name}'
        # path = f'{name}_script'
        os.makedirs(path, exist_ok=True)
        script_filename = f'{path}/{model_name}.sh'
        with open(script_filename, 'a', newline='') as label_file:
            label_file.write(script)
            label_file.write('\n\n')


if __name__ == '__main__':
    file_name = "/home/SEER/result/SEER-best-result.csv"
    result2sh(file_name)
