import torch
import torch.nn as nn
import pickle
import numpy as np


def scale_tensor(tensor,
                 scaler):
    return (tensor - scaler.mean_[0]) / scaler.scale_[0]


def unscale_tensor(tensor,
                   scaler):
    return (tensor * scaler.scale_[0]) + scaler.mean_[0]


class QuantileCombinedModel(nn.Module):
    def __init__(self,
                 device="cpu",
                 input_size=99,
                 num_predictions=3,
                 hidden_size=256,
                 num_layers=2,
                 seed=0,
                 quantiles=None,
                 **kwargs):

        """
        Parameters
        ----------
        device: cpu or cuda
        input_size: number of input sensors
        num_predictions: number of yields to predict
        hidden_size: hidden dimension for lstm
        num_layers: number of layers for the lstm
        seed: random seed for the pytorch
        kwargs: keyword arguments
        """

        super(QuantileCombinedModel, self).__init__()
        if quantiles is None:
            quantiles = [0.95, 0.5, 0.05]
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_predictions = num_predictions
        self.seed = seed
        self.hidden_states_lookback = None
        self.hidden_states_lookahead = None
        self.num_predictions = num_predictions * len(quantiles)
        torch.manual_seed(self.seed)

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bias=True,
            dropout=0.5
        )

        # linear layer to map the hidden encoding to original simulation
        self.linear = nn.Sequential(nn.ReLU(),
                                    nn.Linear(self.hidden_size, self.input_size))

        # linear layer to convert final
        self.linear_predictions = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_size, (self.hidden_size // 2)),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear((self.hidden_size // 2), self.num_predictions)
        )

    def forward(self, x, max_timesteps=325, hidden_states=False):
        # initialize the hidden state and cell state
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.double).to(self.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.double).to(self.device)

        # simulate back
        output, (h_, c_) = self.lstm(x, (h_0, c_0))
        h_, c_ = h_.to(self.device), c_.to(self.device)

        if hidden_states:
            hidden_states_lookback = output

        # map the hidden states to the yields and the sensors
        yields = self.linear_predictions(output)
        sensors_back = self.linear(output)

        # the input to the lookahead will be the last sensor representation from lookback
        decoder_input = sensors_back[:, -1:, :].detach()

        # simulate forward
        for i in range(max_timesteps):
            o, (h_, c_) = self.lstm(decoder_input, (h_, c_))
            cur_yields = self.linear_predictions(o[:, -1:, :])

            if hidden_states:
                if i == 0:
                    hidden_states_lookahead = o
                else:
                    hidden_states_lookahead = torch.cat((hidden_states_lookahead, o), dim=1)

            o = self.linear(o[:, -1:, :])

            if i == 0:
                sensors_forward = o
            else:
                sensors_forward = torch.cat((sensors_forward, o), dim=1)

            yields = torch.cat((yields, cur_yields), dim=1)
            decoder_input = o.detach()

        if hidden_states:
            return hidden_states_lookahead, hidden_states_lookback, yields
        else:
            return sensors_forward, sensors_back, yields

    def load_weights(self, PATH):
        # function for loading the weights from the provided path
        print(f"loading model from epoch: {torch.load(PATH, map_location=self.device)['epoch']}")
        self.load_state_dict(torch.load(PATH, map_location=self.device)["model_state_dict"])
        self.double()
