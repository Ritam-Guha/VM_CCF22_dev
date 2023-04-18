import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import config as config
matplotlib.use('SVG')


def plot_quantile_timesteps(predictions,
                            length,
                            quantiles,
                            yield_list,
                            min_timestep=100,
                            max_timestep=300):
    quantile_prediction = {}
    for q in quantiles:
        quantile_prediction[q] = {}

        for yield_type in yield_list:
            quantile_prediction[q][yield_type] = []

    for i in range(min_timestep, max_timestep + 1):
        for j, yield_type in enumerate(yield_list):
            for k, q in enumerate(quantiles):
                quantile_prediction[q][yield_type].append(
                    predictions[i, j * len(quantiles) + k].detach().numpy())

    for i, yield_type in enumerate(yield_list):
        fig, axes = plt.subplots(1, 1, figsize=(10, 5))
        axes.set_title(f"quantile prediction: {yield_type.replace('yield', 'process_output')}", fontsize=30)
        axes.plot(np.arange(min_timestep, max_timestep + 1), quantile_prediction[0.5][yield_type], color="blue",
                  alpha=0.8,
                  linewidth=3,
                  label="median line")
        axes.fill_between(np.arange(min_timestep, max_timestep + 1),
                          quantile_prediction[quantiles[0]][yield_type],
                          quantile_prediction[quantiles[-1]][yield_type],
                          color="blue", alpha=0.2)
        axes.plot(np.arange(min_timestep, max_timestep + 1),
                  quantile_prediction[quantiles[0]][yield_type],
                  linestyle="dashdot", color="r", label=str(int(quantiles[0] * 100)) + "$^{th}$ quantile",
                  linewidth=3)
        axes.plot(np.arange(min_timestep, max_timestep + 1),
                  quantile_prediction[quantiles[-1]][yield_type],
                  linestyle="dotted", color="g", label=str(int(quantiles[-1] * 100)) + "$^{th}$ quantile",
                  linewidth=3)

        axes.set_xticks(np.arange(min_timestep, max_timestep + 1, 50))
        axes.set_xticklabels([str(z) for z in np.arange(min_timestep, max_timestep + 1, step=50)])
        axes.set_ylabel("prediction", fontsize=20)
        axes.grid()
        axes.legend(loc="lower right")
        fig.savefig(f"{config.BASE_PATH}/static/plots/quantile_yield_simulation/{yield_list[i]}.jpg")


def quantile_sep(yield_sim,
                 quantiles,
                 yield_list):
    # quantile prediction
    pred_yield_data_quantile = {}
    length = yield_sim.shape[0]

    for q in quantiles:
        pred_yield_data_quantile[q] = []

    for q in quantiles:
        q_idx = quantiles.index(q)
        cur_yield_pred = []

        for l in range(length):
            for j in range(len(yield_list)):
                cur_yield_pred.append(yield_sim[l, j * len(quantiles) + q_idx].detach().numpy())
            pred_yield_data_quantile[q].append(cur_yield_pred)

    return pred_yield_data_quantile


def plot_sensors(sensor_readings,
                 sensor_list,
                 gt_timesteps=40):
    num_timesteps, num_sensors = sensor_readings.shape
    for i in range(num_sensors):
        fig, ax = plt.subplots()
        ax.plot(np.arange(gt_timesteps+1), sensor_readings[:gt_timesteps+1, i], color="blue", label="ground_truth")
        ax.plot(np.arange(gt_timesteps, num_timesteps), sensor_readings[gt_timesteps:, i], color="red",
                label="simulation")
        ax.set_title(f"{sensor_list[i]} simulation")
        ax.legend(loc="lower right")
        ax.set_xlabel("timesteps")
        ax.set_ylabel("sensor readings")
        fig.savefig(f"{config.BASE_PATH}/static/plots/sensor_simulation/{sensor_list[i]}.jpg")
        plt.close(fig)


def plot_quantile_yields(yield_readings,
                         yield_list,
                         quantiles=[0.95, 0.5, 0.05]):
    # pred_yield_data_quantile = quantile_sep(yield_sim=yield_readings,
    #                                         quantiles=quantiles,
    #                                         yield_list=yield_list)

    plot_quantile_timesteps(predictions=yield_readings,
                            length=yield_readings.shape[0],
                            quantiles=quantiles,
                            yield_list=yield_list)


def analyze(results):
    plot_sensors(sensor_readings=results["sensor_data"],
                 sensor_list=results["sensor_list"],
                 gt_timesteps=results["gt_steps"])

    plot_quantile_yields(yield_readings=results["yield_data"],
                         yield_list=results["yield_list"],
                         quantiles=results["quantiles"])
