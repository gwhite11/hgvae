import optuna
from gvae_skipcon_attention import VAE, train_vae, train_loader, valid_loader, validate_vae, in_channels, out_channels
import torch


def objective(trial):
    # Hyperparameters to tune
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256])
    dropout_prob = trial.suggest_float("dropout_prob", 0.1, 0.5)

    model = VAE(in_channels, hidden_channels, out_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(10):
        train_loss = train_vae(model, train_loader, optimizer)
        valid_loss = validate_vae(model, valid_loader)

    return valid_loss


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

# Best hyperparameters
print("Best trial:", study.best_trial.params)