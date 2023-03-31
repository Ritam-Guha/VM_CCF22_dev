import vmnet.config as config
from vmnet.utils.path_utils import create_dir, helper_file_naming
from vmnet.models.quantile_combined_model import QuantileCombinedModel
from vmnet.analysis.analyze_model import ModelAnalysis
from vmnet.evaluation.MAD import compute_MAD
from vmnet.evaluation.MAPE import compute_MAPE

import numpy as np
import argparse
import copy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
from captum.attr import IntegratedGradients
import torch

parser = argparse.ArgumentParser("combined")
parser.add_argument("--config_name", type=str,
                    default="base_check",
                    help="path to the model configuration")


class QuantileCombinedAnalysis(ModelAnalysis):
    def __init__(self,
                 device="cpu"):
        super().__init__(type_model="quantile_combined",
                         device=device)
        self.model = None
        self.plot_steps = 200
        self.sensors = None

    def _create_dir_structure(self):
        create_dir(f"analysis/quantile_combined_analysis_results/{self.config_name}")
        for limit in self.config["limits"]:
            create_dir(f"analysis/quantile_combined_analysis_results/{self.config_name}/limit_{limit}")
            create_dir(f"analysis/quantile_combined_analysis_results/{self.config_name}/limit_{limit}/simulated_plots/")

    def get_model(self):
        # load the model
        if "sensors" in self.config.keys():
            self.sensors = self.config["list_sensors"]
        else:
            self.sensors = config.sensors

        input_size = len(self.sensors)
        num_predictions = len(self.config["list_yields"])
        self.model = QuantileCombinedModel(device=self.device,
                                           num_layers=self.config["num_layers"],
                                           input_size=input_size,
                                           hidden_size=self.config["hidden_size"],
                                           num_predictions=num_predictions,
                                           seed=self.config["seed"])

        self.model = self.model.double().to(self.device).eval()
        print(self.model)

    # def load_captum_model(self):
    #     # load the model
    #     if "sensors" in self.config.keys():
    #         self.sensors = self.config["sensors"]
    #     else:
    #         self.sensors = config.sensors
    #     input_size = len(self.sensors)
    #     num_predictions = len(self.config["list_yields"])
    #     self.model = CombinedModel(device="cuda",
    #                                num_layers=self.config["num_layers"],
    #                                input_size=input_size,
    #                                hidden_size=self.config["hidden_size"],
    #                                num_predictions=num_predictions,
    #                                seed=self.config["seed"],
    #                                captum_exploration=True,
    #                                config=self.config)
    #
    #     self.model = self.model.double().to(self.device).eval()
    #     self.load_model()

    def analyze_model(self,
                      config_name,
                      type_data=["test"]):
        print(f"config: {config_name}")
        # initialize
        self.initialize(config_name)
        sensor_data, yield_data, batches, lengths = self.load_data(type_data=type_data)


        # compute error
        yield_data = yield_data.detach().numpy()
        yield_data_scaled = yield_data.copy()
        for idx, yield_type in enumerate(self.config["list_yields"]):
            yield_data[:, idx] = self.yield_scaler[yield_type].inverse_transform(
                yield_data[:, idx].reshape(-1, 1)).squeeze()

        for limit in self.config["limits"]:
            # apply model on the data
            sim_forward, sim_back, yield_sim = self.model(sensor_data[:, :limit, :])
            # yield_sim_scaled = yield_sim.detach().numpy().copy()

            # check sensor sensitivity
            # self.check_model_sensitivity(test_data=sensor_data[:, :limit, :])

            # combine the simulation
            # sim_combined = torch.cat((sim_back, sim_forward), dim=1)
            # self.plot_sensors(orig_sensors=sensor_data,
            #                   sim_sensors=sim_combined,
            #                   batches=batches,
            #                   lengths=lengths,
            #                   limit=limit)

            pred_yield_data = []
            median_idx = config.quantiles.index(0.5)
            for i, l in enumerate(lengths):
                cur_yield_pred = []

                for j in range(len(self.config['list_yields'])):
                    cur_yield_pred.append(yield_sim[i, l - 1, j * len(config.quantiles) + median_idx].detach().numpy())
                pred_yield_data.append(cur_yield_pred)

            pred_yield_data = np.array(pred_yield_data)
            for idx, yield_type in enumerate(self.config["list_yields"]):
                pred_yield_data[:, idx] = self.yield_scaler[yield_type].inverse_transform(
                    pred_yield_data[:, idx].reshape(-1, 1)).squeeze()

            # quantile prediction
            pred_yield_data_quantile = {}
            for q in config.quantiles:
                pred_yield_data_quantile[q] = []

            for i, l in enumerate(lengths):
                for q in config.quantiles:
                    q_idx = config.quantiles.index(q)
                    cur_yield_pred = []

                    for j in range(len(self.config["list_yields"])):
                        cur_yield_pred.append(yield_sim[i, l - 1, j * len(config.quantiles) + q_idx].detach().numpy())
                    pred_yield_data_quantile[q].append(cur_yield_pred)

            for q in config.quantiles:
                pred_yield_data_quantile[q] = np.array(pred_yield_data_quantile[q])
                for idx, yield_type in enumerate(self.config["list_yields"]):
                    pred_yield_data_quantile[q][:, idx] = self.yield_scaler[yield_type].inverse_transform(
                        pred_yield_data_quantile[q][:, idx].reshape(-1, 1)).squeeze()

            # plot timestep quantile
            np.random.seed(0)
            idx = np.random.randint(len(batches))
            self.plot_quantile_timesteps(predictions=yield_sim[idx, :, :],
                                         batch_id=batches[idx],
                                         length=lengths[idx],
                                         gt=yield_data_scaled[idx, :])

            # save the predictions
            prediction_df = pd.DataFrame(np.round(pred_yield_data, 2), columns=self.config["list_yields"])
            for idx, yield_type in enumerate(self.config["list_yields"]):
                prediction_df[f"gt_{yield_type}"] = yield_data[:, idx]
            prediction_df["batches"] = batches
            prediction_df["runtime"] = lengths
            prediction_df.to_csv(
                f"{config.BASE_PATH}/analysis/quantile_combined_analysis_results/{self.config_name}/limit_{limit}/predictions.csv", sep=",",
                index=False)

            # calculate the error
            mad, mape, indiv_mad, indiv_mape = self.calculate_error(pred_yield_data, yield_data)

            # draw error boxplots
            mape_df = pd.DataFrame(indiv_mape, columns=self.config["list_yields"])

            mape_df["runs"] = batches
            self.draw_error_boxplots(mape_df,
                                     save_path=f"{config.BASE_PATH}/analysis/quantile_combined_analysis_results/{self.config_name}/limit_{limit}")

            # store indiv mape
            mape_df.to_csv(f"{config.BASE_PATH}/analysis/quantile_combined_analysis_results/{config_name}/limit_{limit}/indiv_mape.csv",
                           index=False)

            # outlier analysis
            mean_error_wo_outliers = np.zeros(len(self.config["list_yields"]))
            percentage_outliers = np.zeros(len(self.config["list_yields"]))

            for i, cur_yield in enumerate(self.config["list_yields"]):
                new_df, percent_outliers = self.remove_outliers(mape_df, col=cur_yield)
                mean_error_wo_outliers[i] = new_df[cur_yield].mean()
                percentage_outliers[i] = percent_outliers

            metrics = np.round(np.array([mape, mad, mean_error_wo_outliers, percentage_outliers]), decimals=2)
            error_df = pd.DataFrame(metrics, columns=self.config["list_yields"], index=["mape",
                                                                                        "mad",
                                                                                        "error_post_outlier_removal",
                                                                                        "outlier_%"])

            error_df.to_csv(f"{config.BASE_PATH}/analysis/quantile_combined_analysis_results/{self.config_name}/limit_{limit}/error.csv", sep=",",
                            index=True)
            print(error_df)

            # total prediction
            fig_1, fig_2 = self.total_pred(pred_yield_data=yield_sim,
                                           lengths=lengths)
            fig_1.savefig(
                f"{config.BASE_PATH}/analysis/quantile_combined_analysis_results/{self.config_name}/limit_{limit}/total_weight_prediction.jpg", cmap="gray")
            fig_2.savefig(
                f"{config.BASE_PATH}/analysis/quantile_combined_analysis_results/{self.config_name}/limit_{limit}/total_weight_prediction_rand.jpg", cmap="gray")

            # plot quantile
            self.plot_quantile(quantile_preds=pred_yield_data_quantile,
                               orig_yields=yield_data)

        # plot curve
        self.plot_curve()



    @staticmethod
    def calculate_error(yields_preds,
                        yields_orig):
        mape, indiv_mape, classificaition_error = compute_MAPE(yields_orig, yields_preds)
        mad, indiv_mad = compute_MAD(yields_orig, yields_preds)
        return mad, mape, indiv_mad, indiv_mape

    def plot_sensors(self,
                     orig_sensors,
                     sim_sensors,
                     batches,
                     lengths,
                     limit):
        orig_sensors_arr = copy.deepcopy(orig_sensors.detach().numpy())
        sim_sensors_arr = copy.deepcopy(sim_sensors.detach().numpy())
        lengths = copy.deepcopy(lengths)
        batches = copy.deepcopy(batches)

        for i, batch in enumerate(batches):
            create_dir(f"analysis/quantile_combined_analysis_results/{self.config_name}/limit_{limit}/simulated_plots/{batch}")

            for j, sensor in enumerate(self.config["sensors"]):
                fig = plt.figure()
                plt.plot(np.arange(lengths[i]), orig_sensors_arr[i, :lengths[i], j], c="black", label="gt")
                plt.plot(np.arange(lengths[i]), sim_sensors_arr[i, :lengths[i], j], c="r", label="sim")
                plt.title(f"batch: {batch}, simulation sensor: {helper_file_naming(sensor)}")
                plt.xlabel("timesteps")
                plt.ylabel("value")
                plt.legend(loc="upper left")
                plt.savefig(f"{config.BASE_PATH}/analysis/quantile_combined_analysis_results/{self.config_name}/limit_{limit}/simulated_plots/"
                            f"{batch}/{helper_file_naming(sensor)}.jpg", cmap="gray")
                plt.close(fig)

    def total_pred(self,
                   pred_yield_data,
                   lengths,
                   start_step=0,
                   end_step=250):

        fig = plt.figure(figsize=(8, 5))
        pred_yield_data = pred_yield_data.detach().numpy()

        for idx, yield_type in enumerate(self.config["list_yields"]):
            pred_yield_data[:, :, idx] = self.yield_scaler[yield_type].inverse_transform(pred_yield_data[:, :, idx])

        # mean yields
        yield_progress = pred_yield_data.sum(axis=2)
        yield_mean = yield_progress.mean(axis=0)[start_step:(end_step + 1)]
        yield_l = yield_progress.min(axis=0)[start_step:(end_step + 1)]
        yield_h = yield_progress.max(axis=0)[start_step:(end_step + 1)]

        # plot the progress of yields
        fig = plt.figure()
        plt.plot(np.arange(end_step - start_step + 1), yield_mean, c="red", label="total weight")
        plt.fill_between(np.arange(end_step - start_step + 1), yield_l, yield_h, alpha=0.3, edgecolor="red",
                         facecolor="red")
        plt.xticks(np.arange(0, end_step - start_step + 1, 50),
                   labels=[str(x) for x in np.arange(start_step, end_step + 1, 50)])
        plt.xlabel("timesteps")
        plt.ylabel("yield prediction")
        plt.title(f"total weight progress")
        plt.legend(loc="upper right")
        plt.savefig(f"{config.BASE_PATH}/analysis/quantile_combined_analysis_results/{self.config_name}/progress_total_weight.jpg", cmap="gray")

        # plot some batches
        n_batches = 5
        fig_2 = plt.figure()
        np.random.seed(5)
        colors = cm.rainbow(np.linspace(0, 1, n_batches))
        rand_batch = list(np.random.randint(low=0, high=yield_progress.shape[0] - 1, size=(1, n_batches)).squeeze())

        for idx, i in enumerate(rand_batch):
            plt.plot(np.arange(lengths[idx]), yield_progress[i, :lengths[idx]], c=colors[idx], label=f"batch_{idx}",
                     alpha=0.7)

        plt.xticks(np.arange(0, end_step - start_step + 1, 50),
                   labels=[str(x) for x in np.arange(start_step, end_step + 1, 50)])
        plt.xlabel("timesteps")
        plt.ylabel("yield prediction")
        plt.title(f"total weight progress")
        plt.legend(loc="lower right")
        # plt.show()

        return fig, fig_2

    def plot_curve(self):
        epoch_wise_loss = pickle.load(open(
            f"{config.BASE_PATH}/model_storage/quantile_combined/{self.config_name}/epoch_wise_loss.pickle",
            "rb"))
        loss = {}
        type_exps = ["train", "val", "test"]
        color_code = {"train": "red",
                      "val": "blue",
                      "test": "black"}

        for type_exp in type_exps:
            loss[type_exp] = epoch_wise_loss[type_exp]

        # plot the mape
        yield_mape_list = {"train": [],
                           "val": [],
                           "test": []}

        # plot total loss
        total_loss_list = {"train": [],
                           "val": [],
                           "test": []}

        for i in range(len(loss["train"])):
            for type_exp in type_exps:
                yield_mape_list[type_exp].append(loss[type_exp][i]["separate_yield_mape"])
                total_loss_list[type_exp].append(loss[type_exp][i]["total_loss"])

        for type_exp in type_exps:
            yield_mape_list[type_exp] = np.array(yield_mape_list[type_exp])

        for i, yield_type in enumerate(self.config["list_yields"]):
            fig = plt.figure()
            for type_exp in type_exps:
                plt.plot(np.arange(len(loss[type_exp])), yield_mape_list[type_exp][:, i], label=type_exp,
                         c=color_code[type_exp])
            plt.legend(loc="upper right")
            plt.title(f"{yield_type} training")
            plt.savefig(f"{config.BASE_PATH}/analysis/quantile_combined_analysis_results/{self.config_name}/limit_{self.config['limits'][0]}/"
                        f"mape_{yield_type}_training_curve", cmap="gray")
            plt.close(fig)

        # total loss
        fig = plt.figure()
        for type_exp in type_exps:
            plt.plot(np.arange(len(loss[type_exp])), total_loss_list[type_exp], label=type_exp,
                     c=color_code[type_exp])
        plt.legend(loc="upper right")
        plt.title(f"training curves total loss")
        plt.savefig(f"{config.BASE_PATH}/analysis/quantile_combined_analysis_results/{self.config_name}/limit_{self.config['limits'][0]}/"
                    f"loss_training_curve", cmap="gray")
        plt.close(fig)

    def check_model_sensitivity(self,
                                test_data):
        self.load_captum_model()
        self.model = self.model.to("cuda").train()
        test_data = test_data.to("cuda")
        ig = IntegratedGradients(self.model)
        sensitivity_matrix = np.zeros((len(self.config["list_yields"]), len(config.sensors)))
        for i, target in enumerate(self.config["list_yields"]):
            attribution = ig.attribute(test_data, target=i)
            mean_attribution = attribution.mean(dim=0).mean(dim=0).detach().cpu().numpy()
            sensitivity_matrix[i, :] = mean_attribution

        for i in range(sensitivity_matrix.shape[0]):
            sensitivity_matrix[i, :] = (sensitivity_matrix[i, :] - np.min(sensitivity_matrix[i, :])) / \
                                       (np.max(sensitivity_matrix[i, :]) - np.min(sensitivity_matrix[i, :]))
        self.plot_sensitivity_matrix(sensitivity_matrix,
                                     self.config["list_yields"],
                                     config.sensors,
                                     self.config_name)

    def plot_sensitivity_matrix(self,
                                sensitivity_matrix,
                                list_yields,
                                list_sensors,
                                config_name):

        num_sensors_per_plot = 15

        for i, yield_type in enumerate(list_yields):
            cur_sensitivity_info = sensitivity_matrix[i, :]
            sorted_idx = np.argsort(-cur_sensitivity_info)
            cur_sensitivity_info = cur_sensitivity_info[sorted_idx][:num_sensors_per_plot]
            cur_list_sensors = [list_sensors[j] for j in sorted_idx][:num_sensors_per_plot]

            fig, ax = plt.subplots()
            ax.bar(cur_list_sensors, cur_sensitivity_info)
            ax.set_xticklabels(cur_list_sensors, rotation=90)
            ax.set_xlabel("sensors")
            ax.set_ylabel("normalized sensitivity")
            ax.set_title(f"sensitivity analysis for {yield_type}")
            fig.savefig(f"{config.BASE_PATH}/analysis/quantile_combined_analysis_results/{config_name}/limit_{self.config['limits'][0]}/sensitive_sensors_yield_{yield_type}.jpg",
                        tight_layout=True,
                        bbox_inches='tight', cmap="gray")

        # for j in range(int(np.ceil(sensitivity_matrix.shape[1] / num_sensors_per_plot))):
        #     start_idx = j * num_sensors_per_plot
        #     end_idx = min(start_idx + num_sensors_per_plot, sensitivity_matrix.shape[1])
        #     num_sensors = (end_idx - start_idx)
        #
        #     fig, axes = plt.subplots(figsize=(12, 8))
        #     colors = cm.rainbow(np.linspace(0, 1, len(list_yields)))
        #     for i in range(sensitivity_matrix.shape[0]):
        #         axes.scatter(np.arange(num_sensors), np.ones(num_sensors) * (i + 1),
        #                      s=sensitivity_matrix[i, start_idx:end_idx] * 1000,
        #                      c=colors[i], label=list_yields[i], edgecolors="black")
        #
        #     axes.set_xticks(np.arange(num_sensors), labels=list_sensors[start_idx:end_idx])
        #     axes.tick_params(axis='x', rotation=90)
        #     axes.set_xlabel("sensors")
        #     axes.set_ylabel("yields")
        #     axes.set_title("Sensitivity Analysis for the Sensors")
        #     axes.set_yticks(np.arange(len(list_yields)) + 1, labels=list_yields)
        #     fig.savefig(f"{config.BASE_PATH}/analysis/quantile_combined_analysis_results/{config_name}/limit_{self.config['limits'][0]}/sensitive_sensors_{j}.jpg",
        #                 tight_layout=True,
        #                 bbox_inches='tight')
        #     fig.show()
        # print(sensitivity_matrix)

    def plot_quantile(self,
                      quantile_preds,
                      orig_yields):

        for i, yield_type in enumerate(self.config["list_yields"]):
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(np.arange(orig_yields.shape[0]), orig_yields[:, i], edgecolors="black", color="blue")
            ax.fill_between(np.arange(orig_yields.shape[0]),
                            quantile_preds[min(config.quantiles)][:, i],
                            quantile_preds[max(config.quantiles)][:, i],
                            color="blue",
                            alpha=0.4)
            bool_decision = np.array(orig_yields[:, i] >= quantile_preds[min(config.quantiles)][:, i]) * \
                            np.array(orig_yields[:, i] <= quantile_preds[max(config.quantiles)][:, i])
            accuracy = np.round(sum(bool_decision)/orig_yields.shape[0] * 100, 2)
            ax.set_xlabel("batches")
            ax.set_ylabel("prediction")
            ax.set_title(f"quantile prediction for {yield_type.replace('yield', 'process_output')}")
            ax.annotate(f"accuracy: {accuracy}",
                        xy=(150, min(quantile_preds[min(config.quantiles)][:, i])), xycoords='data',
                        xytext=(150, min(quantile_preds[min(config.quantiles)][:, i])), textcoords='data')
            fig.savefig(f"{config.BASE_PATH}/analysis/quantile_combined_analysis_results/{self.config_name}/limit_{self.config['limits'][0]}"
                        f"/quantile_pred_{yield_type}.jpg",
                        tight_layout=True,
                        bbox_inches='tight', cmap="gray")
            fig.show()

    def plot_quantile_timesteps(self,
                                predictions,
                                batch_id,
                                length,
                                gt,
                                min_timestep=100,
                                max_timestep=300):

        quantile_prediction = {}
        for q in config.quantiles:
            quantile_prediction[q] = {}

            for yield_type in self.config["list_yields"]:
                quantile_prediction[q][yield_type] = []

        for i in range(min_timestep, max_timestep + 1):
            for j, yield_type in enumerate(self.config["list_yields"]):
                for k, q in enumerate(config.quantiles):
                    quantile_prediction[q][yield_type].append(predictions[i, j * len(config.quantiles) + k].detach().numpy())

        error = {}
        for j, yield_type in enumerate(self.config["list_yields"]):
            error[yield_type] = torch.abs(torch.DoubleTensor(np.array(quantile_prediction[0.5][yield_type])) - gt[j]) / gt[
                j] * 100

        for i, yield_type in enumerate(self.config["list_yields"]):
            fig, axes = plt.subplots(2, 1, figsize=(13, 10), gridspec_kw={'height_ratios': [3, 1]})
            # axes[0].plot(np.arange(min_timestep, max_timestep + 1), error[yield_type], color="black")
            # axes[0].grid()
            # axes[0].set_ylabel("error")
            axes[0].set_title(f"quantile prediction: {yield_type.replace('yield', 'process_output')}", fontsize=30)
            axes[0].plot(np.arange(min_timestep, max_timestep + 1), quantile_prediction[0.5][yield_type], color="blue",
                         alpha=0.8,
                         linewidth=3,
                         label="median line")
            axes[0].fill_between(np.arange(min_timestep, max_timestep + 1),
                                 quantile_prediction[config.quantiles[0]][yield_type],
                                 quantile_prediction[config.quantiles[-1]][yield_type],
                                 color="blue", alpha=0.2)
            axes[0].plot(np.arange(min_timestep, max_timestep + 1),
                         quantile_prediction[config.quantiles[0]][yield_type],
                         linestyle="dashdot", color="r", label=str(int(config.quantiles[0] * 100)) + "$^{th}$ quantile", linewidth=3)
            axes[0].plot(np.arange(min_timestep, max_timestep + 1),
                         quantile_prediction[config.quantiles[-1]][yield_type],
                         linestyle="dotted", color="g", label=str(int(config.quantiles[-1] * 100)) + "$^{th}$ quantile", linewidth=3)
            axes[0].vlines(length - 1, quantile_prediction[config.quantiles[-1]][yield_type][0], gt[i],
                           linestyle="dashed", color="black")
            axes[0].hlines(gt[i], min_timestep, length - 1, linestyle="dashed", color="black")
            axes[0].scatter(length - 1, gt[i], color="black", label="ground truth")
            axes[0].set_xticks(np.arange(min_timestep, max_timestep + 1, 50))
            axes[0].set_xticklabels([str(z) for z in np.arange(min_timestep, max_timestep + 1, step=50)])
            axes[0].set_ylabel("prediction", fontsize=20)
            axes[0].grid()
            axes[0].legend(loc="lower right")
            # print(np.arange(min_timestep, max_timestep + 1).shape)
            # print((np.array(quantile_prediction[config.quantiles[0]][yield_type]) - np.array(quantile_prediction[config.quantiles[-1]][yield_type])).shape)
            axes[1].plot(np.arange(min_timestep, max_timestep + 1), np.array(quantile_prediction[config.quantiles[0]][yield_type]) - np.array(quantile_prediction[config.quantiles[-1]][yield_type]), color="black")
            axes[1].grid()
            axes[1].set_ylabel("quantile width", fontsize=20)
            axes[1].set_xlabel("time steps", fontsize=20)
            fig.savefig(
                f"{config.BASE_PATH}/analysis/quantile_combined_analysis_results/{self.config_name}/limit_{self.config['limits'][0]}"
                f"/quantile_timeline_pred_{yield_type}.jpg",
                tight_layout=True,
                bbox_inches='tight', cmap="gray")
            plt.show()






def main():
    args = parser.parse_args()
    config_name = args.config_name
    combined_analysis = QuantileCombinedAnalysis()
    combined_analysis.analyze_model(config_name=config_name,
                                    type_data=["test"])


if __name__ == "__main__":
    main()
