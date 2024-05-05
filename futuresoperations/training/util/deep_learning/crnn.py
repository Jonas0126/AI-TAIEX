import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, input_size, feature_size,hidden_size, output_size, n_layers=1,dropout=0.3):
        super(CRNN, self).__init__()
        self.input_size = input_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.c1 = nn.Conv1d(input_size, feature_size, kernel_size = 5, stride = 5)
        self.lstm = nn.LSTM(feature_size, hidden_size, n_layers, batch_first=True,dropout=dropout)
        self.leakyRelu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):

        # Turn (batch_size x seq_len x input_size) into (batch_size x input_size x seq_len) for CNN
        inputs = inputs.transpose(1, 2)

        # Run through Conv1d and Pool1d layers
        p = self.c1(inputs)
        p = self.leakyRelu(p)

        # Turn (batch_size x hidden_size x seq_len) back into (seq_len x batch_size x hidden_size) for RNN
        p = p.permute(0,2,1)

        # h0 = torch.zeros(self.n_layers, p.size(0), self.hidden_size).requires_grad_().to(device)
        # c0 = torch.zeros(self.n_layers, p.size(0), self.hidden_size).requires_grad_().to(device)
        # out, (hn, cn) = self.lstm(p, (h0.detach(), c0.detach()))
        out, hidden = self.lstm(p)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])

        return out