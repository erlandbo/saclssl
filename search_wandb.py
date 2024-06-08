import wandb
from main_sacl import main_train


def sweep_hparams():
    # Define sweep config
    cosine_sweep_configuration = {
        "method": "random",
        "name": "sweep",
        "metric": {"goal": "maximize", "name": "val_acc"},
        "parameters": {
            "s_init_t": {"values": [0.0, -1.0, -1.75, -2.0, -2.75, -3.0, -3.75, -4.0]},
            "alpha": {"values": [0.125, 0.25, 0.5]},
            "rho_const": {"values": [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]},
        },
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 3,
            "eta": 2,
            "strict": True,
        }
    }

    prob_metric_sweep_configuration = {
        "method": "random",
        "name": "sweep",
        "metric": {"goal": "maximize", "name": "val_acc"},
        "parameters": {
            "s_init_t": {"values": [-1, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]},
            "alpha": {"values": [0.125, 0.25, 0.5]},
            "rho_const": {"values": [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]},
        },
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 3,
            "eta": 2,
            "strict": True,
        }
    }

    sweep_configuration = cosine_sweep_configuration

    # Initialize sweep by passing in config.
    # (Optional) Provide a name of the project.
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="scl_tinyim_sweep_final_final")
    wandb.agent(sweep_id, function=main_train, count=200)


if __name__ == "__main__":
    # Use same arguments as in main_sacl.py and include argument --sweep_hparams
    sweep_hparams()

