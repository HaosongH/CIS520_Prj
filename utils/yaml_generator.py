import yaml

LRs = [0.005, 0.10, 0.15, 0.20]
weight_decays = [0.0, 0.1]
scheduler_gammas = [0.95, 0.9, 0.85, 0.8]
kld_weights = [0.00025, 0.0005, 0.00075, 0.001]
recon_weights = [100, 150, 200]
n_epochs = [5, 10, 20, 30]
models = ["VanillaVAE"]
for LR in LRs:
    for weight_decay in weight_decays:
        for scheduler_gamma in scheduler_gammas:
            for kld_weight in kld_weights:
                for recon_weight in recon_weights:
                    for n_epoch in n_epochs:
                        for model in models:
                            config = {'model_params' : {
                                        'name':"VanillaVAE",
                                        'in_channels': 3,
                                        'hidden_dims': None,
                                        'latent_dim':128
                                    },
                                    'data_params' : {
                                        "data_path":"Data/",
                                        "train_batch_size": 64,
                                        "val_batch_size":64,
                                        "patch_size": 64,
                                        "num_worker": 4
                                    },
                                    'exp_params' : {
                                        "LR":LR,
                                        "weight_decay":weight_decay,
                                        "scheduler_gamma":scheduler_gamma,
                                        "kld_weight":kld_weight,
                                        "recon_weight":recon_weight,
                                        "manual_seed":1265
                                    },
                                    'trainer_params': {
                                        "gpus": [1],
                                        "n_epochs":n_epoch
                                    },
                                    'logging_params' : {
                                        "save_dir":"logs/",
                                        "name":model
                                    }}
                                    

                            with open(f'config/exp_{LR}_{weight_decay}_{kld_weight}_{recon_weight}_trainer_{n_epoch}_log_{model}.yaml', 'w') as file:
                                yaml.dump(config, file)
