from .quantile_combined_model import QuantileCombinedModel
from .analysis import analyze

import torch
import pandas as pd
import pickle

sensor_scaler = pickle.load(open(f"codes/sensor_scaler_obj.pickle", "rb"))
quantiles = [0.95, 0.5, 0.05]


def prepare_data(data_file):
    data = pd.read_csv(data_file)
    data = data.iloc[:40, :].drop(columns=["timestamp"])
    data_cols = list(data.columns)
    data = sensor_scaler.transform(data)
    data_tensor = torch.Tensor(data).type(torch.DoubleTensor)
    data_tensor = data_tensor.unsqueeze(0)

    return data_tensor, data_cols


def run(file_name):
    model = QuantileCombinedModel().eval()
    model.load_weights(f"codes/training_best.pt")
    data, sensor_names = prepare_data(f"codes/data/{file_name}")
    gt_steps = data.shape[1]
    sensor_forward, _, yields = model(data)
    sensor_data = torch.cat((data, sensor_forward), dim=1).squeeze().detach()
    yields = yields.detach().squeeze()

    yield_list = [f"yield_{i}" for i in range(3)]

    results = {
        "sensor_data": sensor_data,
        "yield_data": yields,
        "sensor_list": sensor_names,
        "gt_steps": gt_steps,
        "quantiles": quantiles,
        "yield_list": yield_list
    }

    analyze(results)


def main():
    run()


if __name__ == "__main__":
    main()
