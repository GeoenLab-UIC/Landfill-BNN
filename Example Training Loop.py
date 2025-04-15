# Initialize the model, loss function, and optimizer
input_size = X_train.shape[1]
hidden_size = 20
activation_1 = nn.ReLU
activation_2 = nn.Sigmoid
model = BNN(input_size, hidden_size, 2, activation_1, activation_2)

# Switch to Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
criterion = nn.MSELoss()


# Training loop
total_epochs = 4000
model.train()
for epoch in range(total_epochs):
    loss = train_step(n_samples =100)
    
    if epoch % 200 == 0:
        with torch.no_grad():
            Y_pred_train = model(X_train_tensor)
            Y_pred_test = model(X_test_tensor)
            train_loss = mean_squared_error(Y_train_tensor.numpy(), Y_pred_train.numpy())
            test_loss = mean_squared_error(Y_test_tensor.numpy(), Y_pred_test.numpy())
            print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
