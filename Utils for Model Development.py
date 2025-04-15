
def mc_predictions(model, X_tensor, n_samples=100):
    model.train()
    predictions = torch.stack([model(X_tensor) for _ in range(n_samples)])
    mean_prediction = predictions.mean(dim=0)
    lower_ci = predictions.quantile(0.005, dim=0)
    upper_ci = predictions.quantile(0.995, dim=0)
    std_ci= predictions.std(dim=0)
    return mean_prediction, std_ci, lower_ci, upper_ci

def train_step(n_samples):
    # Zero the gradients from the previous step
    optimizer.zero_grad()
    
    
    # Forward pass
    pred, _, _, _ = mc_predictions(model, X_train_tensor, n_samples = n_samples)
    
    mse_loss = criterion(pred, Y_train_tensor)
    kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    
    
    # Compute the loss
    loss = mse_loss + 0.01*kl_loss(model)
    
    # Backward pass to compute gradients
    loss.backward()
    
    # Update the weights
    optimizer.step()
    
    return loss
