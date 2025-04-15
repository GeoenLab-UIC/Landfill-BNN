class Spike_and_Slab_BNN_Layer(nn.Module):
    """
    Bayesian Input Layer with Spike-and-Slab Prior for Weights and Gaussian for Biases.
    """
    def __init__(self, input_dim, output_dim, rho_prior=-0.432, lambda0=0.99,
                 activation_fn=nn.ReLU):  # Add activation function as parameter
        super(MLPLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Initialize mu, rho, and theta parameters for weights (spike-and-slab)
        self.w_mu = nn.Parameter(torch.Tensor(input_dim, output_dim).uniform_(-0.31, 0.31))
        self.w_rho = nn.Parameter(torch.Tensor(input_dim, output_dim).uniform_(rho_prior, rho_prior))
        self.w_theta = nn.Parameter(torch.logit(torch.Tensor(input_dim, output_dim).uniform_(lambda0, lambda0)))

        # Initialize mu and rho for Bayesian biases (Gaussian)
        self.b_mu = nn.Parameter(torch.Tensor(output_dim).uniform_(-0.31, 0.31))
        self.b_rho = nn.Parameter(torch.Tensor(output_dim).uniform_(rho_prior, rho_prior))

        self.rho_prior = rho_prior
        self.kl = 0  # KL divergence storage

        # Set the activation function dynamically
        self.activation_fn = activation_fn()

    def forward(self, X, temp=1.0, phi_prior=0.5):
        """
        Forward pass with Monte Carlo sampling for weights and biases.
        """
        # Convert rho to standard deviation
        sigma_w = torch.log(1 + torch.exp(self.w_rho))
        sigma_b = torch.log(1 + torch.exp(self.b_rho))
        sigma_prior = torch.log(1 + torch.exp(torch.tensor(self.rho_prior)))

        # Sample binary gates using Gumbel-Softmax (for weights only)
        u_w = torch.rand(self.w_theta.shape)
        gamma_w = F.gumbel_softmax(self.w_theta, tau=temp, hard=False)

        # Sample weights and biases with Gaussian noise
        epsilon_w = Normal(0, 1).sample(self.w_mu.shape)
        epsilon_b = Normal(0, 1).sample(self.b_mu.shape)
        w = gamma_w * (self.w_mu + sigma_w * epsilon_w)
        b = self.b_mu + sigma_b * epsilon_b  # Gaussian biases

        # Compute output
        output = X @ w + b

        # Apply the activation function to the output
        output = self.activation_fn(output)

        # Compute KL divergence for weights (spike-and-slab) and biases (Gaussian)
        w_phi = torch.sigmoid(self.w_theta)
        kl_w = w_phi * (torch.log(w_phi) - torch.log(torch.tensor(phi_prior))) + \
               (1 - w_phi) * (torch.log(1 - w_phi) - torch.log(1 - torch.tensor(phi_prior))) + \
               w_phi * (torch.log(sigma_prior) - torch.log(sigma_w) +
                        0.5 * ((sigma_w ** 2 + self.w_mu ** 2) / sigma_prior ** 2) - 0.5)

        kl_b = torch.log(sigma_prior) - torch.log(sigma_b) + \
               0.5 * ((sigma_b ** 2 + self.b_mu ** 2) / sigma_prior ** 2) - 0.5

        self.kl = torch.sum(kl_w) + torch.sum(kl_b)  # Store KL divergence

        return output

