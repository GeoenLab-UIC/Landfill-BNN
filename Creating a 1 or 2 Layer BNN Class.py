class BNN(nn.Module):
    def __init__(self, input_size=10, hidden_size_1=15, num_layers=1, 
                 activation_1=nn.ReLU, activation_2=nn.ReLU):
        super().__init__()
        self.num_layers = num_layers
        self.activation_1 = activation_1()
        self.activation_2 = activation_2() if activation_2 else None

        self.bnn1 = bnn.BayesLinear(0, 0.5, input_size, hidden_size_1)
        if num_layers == 2:
            self.bnn2 = bnn.BayesLinear(0, 0.5, hidden_size_1, hidden_size_1)
            self.bnn_out = bnn.BayesLinear(0, 0.5, hidden_size_1, 1)
        else:
            self.bnn_out = bnn.BayesLinear(0, 0.5, hidden_size_1, 1)

    def forward(self, x):
        x = self.activation_1(self.bnn1(x))
        if self.num_layers == 2:
            x = self.activation_2(self.bnn2(x))
        return self.bnn_out(x)
