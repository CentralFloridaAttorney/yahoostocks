from __future__ import print_function

import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from python.yahoostocks.yahoostock import YahooStock
from sklearn.preprocessing import MinMaxScaler

matplotlib.use('Agg')


class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.size_layer1 = 2
        self.size_layer2 = 16
        self.size_layer3 = 1
        self.lstm1 = nn.LSTMCell(self.size_layer1, self.size_layer1)
        self.lstm2 = nn.LSTMCell(self.size_layer1, self.size_layer2)
        self.lstm3 = nn.LSTMCell(self.size_layer2, self.size_layer3)
        self.linear = nn.Linear(self.size_layer3, self.size_layer3)

    def forward(self, _forward_input, future_range=0):
        outputs = []
        # _forward_input = torch.te(_forward_input)
        h_t = torch.zeros(_forward_input.size(0), self.size_layer1, dtype=torch.double)
        c_t = torch.zeros(_forward_input.size(0), self.size_layer1, dtype=torch.double)
        h_t2 = torch.zeros(_forward_input.size(0), self.size_layer2, dtype=torch.double)
        c_t2 = torch.zeros(_forward_input.size(0), self.size_layer2, dtype=torch.double)
        h_t3 = torch.zeros(_forward_input.size(0), self.size_layer3, dtype=torch.double)
        c_t3 = torch.zeros(_forward_input.size(0), self.size_layer3, dtype=torch.double)
        for input_t in _forward_input.split(self.size_layer1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3, c_t3))
            output = self.linear(h_t3)
            outputs += [output]

        outputs = torch.cat(outputs, dim=1)
        return outputs


class TrainerMain:
    def __init__(self):
        # init the dataset
        self.this_ticker = 'MSFT'
        stock_item = YahooStock(self.this_ticker)
        stock_item._drop_column(5)
        classification = stock_item.get_classification_greater_prior(5, 2)
        close_price = stock_item.price_frame['close']
        volume = stock_item.price_frame['volume']
        stock_item._drop_column(0)
        stock_item._add_column(close_price)
        stock_item._add_column(volume)
        stock_item._add_column(classification)
        # self.stock_price_data = stock_item.add_column(classification).astype(dtype=float)
        # try to scale data
        min_max = MinMaxScaler()

        self.stock_price_data = min_max.fit_transform(stock_item.data_frame)
        self.x_train, self.x_target, self.y_train, self.y_target = stock_item.get_test_train_split(
            _data=self.stock_price_data, _train_end_col=2, _batch_size=3, _train_ratio=.6, _target_column_start=3)
        self.x_train = torch.from_numpy(self.x_train)
        self.y_train = torch.from_numpy(self.y_train)
        self.x_target = torch.from_numpy(self.x_target)
        self.y_target = torch.from_numpy(self.y_target)

        # init the optimizer
        parser = argparse.ArgumentParser()
        parser.add_argument('--steps', type=int, default=25, help='steps to run')
        self.opt = parser.parse_args()

        # set random seed to 0
        numpy.random.seed(0)
        torch.manual_seed(0)
        self.criterion = nn.MSELoss(reduction='mean')

        self.seq = Sequence()
        self.seq.double()
        self.optimizer = optim.LBFGS(self.seq.parameters(), lr=0.0001)

    @property
    def run_trainer(self):
        # begin to train
        for i in range(self.opt.steps):
            print('STEP: ', i)

            def closure():
                self.optimizer.zero_grad()
                # _x_train_clone = torch.clone(self.x_train)
                _out = self.seq(self.x_train)
                _loss = self.criterion(_out, self.x_target)
                print('loss:', _loss.item())
                _loss.backward()
                return _loss

            self.optimizer.step(closure)
            # begin to predict, no need to track gradient here
            with torch.no_grad():
                _y_train_clone = torch.clone(self.y_train)
                pred = self.seq(_y_train_clone)
                loss = self.criterion(pred, self.y_target)
                # loss = self.criterion()
                print('test loss:', loss.item())
                y = pred.detach().numpy()
            # draw the result
            plt.figure(figsize=(30, 10))
            plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
            plt.xlabel('x', fontsize=20)
            plt.ylabel('y', fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)

            def draw(yi, color):
                _size_x = len(self.x_train)
                _size_y = len(yi)
                # plt.plot(yi, linewidth=2.0, scalex=_size_x, scaley=_size_y)
                # plt.plot(self.y_target, linewidth=2.0, scalex=_size_x, scaley=_size_y)
                plt.plot(yi, linewidth=2.0, scalex=_size_x, scaley=_size_y)

                # plt.plot(yi, color + ':', linewidth=2.0, scalex=True, scaley=True)

            _y_val = y
            draw(y, 'r')
            # draw(y[1], 'g')
            # draw(y[2], 'b')
            plt.savefig('../../data/pdf/predict%d.pdf' % i)
            plt.close()


trainer = TrainerMain()
trainer.run_trainer

print('Finished')
