import copy
import os
import pickle
from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml

import vmnet.config as config
from vmnet.data_utils.data_loaders.data_collection import collect_data
from vmnet.utils.helper_functions import embed_tsne, cluster_kmeans
from vmnet.utils.path_utils import helper_file_naming
plt.rcParams.update({'font.size': 15})


class ModelAnalysis:
    def __init__(self,
                 type_model,
                 device="cpu"):
        self.type_model = type_model
        self.config_name = None
        self.config = None
        self.model_id = None
        self.cur_model_storage_dir = None
        self.model = None
        self.device = device
        self.sensor_scaler = None
        self.yield_scaler = None

    def initialize(self,
                   config_name):
        # get the config and load the model
        self.config_name = config_name
        self.config = self.load_config()
        self._create_dir_structure()

        self.sensor_scaler = pickle.load(
            open(f"{config.BASE_PATH}/data/data_partition/set_0/sensor_scaler_obj.pickle", "rb"))
        self.yield_scaler = pickle.load(
            open(f"{config.BASE_PATH}/data/data_partition/set_0/yield_scaler_obj.pickle", "rb"))

        self.cur_model_storage_dir = f"model_storage/{self.type_model}/{self.config_name}"
        self.get_model()
        self.load_model()

    def load_data(self,
                  type_data=["test"]):

        list_yields = None
        if "list_yields" in self.config.keys():
            list_yields = self.config["list_yields"]
        else:
            list_yields = ["final_weight"]

        sensor_data, yield_data, batches, lengths = collect_data(type_data=type_data,
                                                                 scale=self.config['scale'],
                                                                 list_sensors=self.sensors,
                                                                 list_yields=list_yields)

        return sensor_data, yield_data, batches, lengths

    def load_config(self):
        return yaml.load(open(f"{config.BASE_PATH}/training_configs/{self.type_model}/{self.config_name}.yaml"),
                         Loader=yaml.FullLoader)

    def load_model(self):
        # if weights exist, load them
        if os.path.isdir(f"{config.BASE_PATH}/{self.cur_model_storage_dir}"):
            if os.path.exists(f"{config.BASE_PATH}/{self.cur_model_storage_dir}/training_best.pt"):
                print("found the old weights.. loading them..")
                self.model.load_weights(f"{config.BASE_PATH}/{self.cur_model_storage_dir}/training_best.pt")

    def plot_reconstruction(self,
                            recon_data,
                            sensor_data):
        mean_recon_data = recon_data.mean(axis=0)
        mean_sensor_data = sensor_data.mean(axis=0)

        unscaled_recon_data = self.sensor_scaler.inverse_transform(mean_recon_data)
        unscaled_sensor_data = self.sensor_scaler.inverse_transform(mean_sensor_data)

        for i, sensor in enumerate(config.sensors):
            fig = plt.figure(figsize=(8, 5))
            plt.plot(np.arange(self.plot_steps), unscaled_recon_data[:self.plot_steps, i],
                     label="reconstructed values", color="blue")
            plt.plot(np.arange(self.plot_steps), unscaled_sensor_data[:self.plot_steps, i],
                     label="ground truth values", color="red")
            plt.title(f"reconstructed plots for {sensor}")
            plt.xlabel("time steps")
            plt.ylabel("sensor readings")
            plt.savefig(f"{config.BASE_PATH}/analysis/{self.type_model}_results/{self.config_name}/reconstructed_plots/"
                        f"{helper_file_naming(sensor)}.jpg", tight_layout=True)
            plt.close(fig)

    def cluster_hidden(self,
                       hidden_arr,
                       batches):
        # cluster the embedded data
        embedded_arr = embed_tsne(hidden_arr, n_components=2)
        labels = cluster_kmeans(hidden_arr, n_clusters=3)
        df = pd.DataFrame(embedded_arr, columns=["component_1", "component_2"])
        df["label"] = labels
        df["label"] = df["label"].astype(str)
        df["batch"] = batches

        fig = px.scatter(df, x="component_1", y="component_2", color="label", hover_data=["batch"],
                         title=f"{self.config_name} clusters")
        fig.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')),
                          selector=dict(mode='markers'))
        fig.write_image(
            f"{config.BASE_PATH}/analysis/{self.type_model}_results/{self.config_name}/clustered_batches.jpg")
        fig.write_html(
            f"{config.BASE_PATH}/analysis/{self.type_model}_results/{self.config_name}/clustered_batches.html")
        fig.show()

    @staticmethod
    def draw_error_boxplots(df,
                            save_path):
        runs = [f"batch={run}" for run in df["runs"].values]
        yield_list = list(set(config.yields).intersection(df.columns))
        df = df[config.yields]
        print(df.mean(axis=0))
        print(yield_list)
        labels = [yield_type.replace("yield", "process_output") for yield_type in yield_list]
        print(labels)

        fig, ax = plt.subplots(figsize=(12, 10))
        # notch shape box plot
        boxplot = ax.boxplot(df.to_numpy(),
                             notch=True,  # notch shape
                             vert=True,  # vertical box alignment
                             patch_artist=True,  # fill with color
                             labels=labels)  # will be used to label x-ticks
        for median in boxplot['medians']:
            median.set_color('black')
        plt.ylabel("error")
        plt.xticks(np.arange(1, len(config.yields)+1), labels, rotation=20)

        # fill with colors
        colors = ['blue', 'green', 'red']
        for patch, color in zip(boxplot['boxes'], colors):
            patch.set(alpha=0.5, facecolor=color)

        plt.savefig(f"{save_path}/error_dist.jpg", dpi=400)
        plt.show()

    @staticmethod
    def remove_outliers(df,
                        col):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        new_df = copy.deepcopy(df)
        new_df = new_df[~((new_df[col] < (Q1 - 1.5 * IQR)) | (new_df[col] > (Q3 + 1.5 * IQR)))]
        percentage_outliers = ((df.shape[0] - new_df.shape[0]) / df.shape[0]) * 100

        return new_df, percentage_outliers

    @abstractmethod
    def analyze_model(self,
                      type_data,
                      config_name):
        pass

    @abstractmethod
    def _create_dir_structure(self):
        pass

    @abstractmethod
    def get_model(self):
        pass
