"""Model config in json format"""

"""Model config in json format"""

CFG = {
    "data": {
        "path": None,
    },
    "model": {
        "var_model": {
            "fh": 5
        },
        "tft_model": {
          'epochs': 100,
          'dropout': 0.70,
          'optimizer': 'ranger',
          'batch_size': 64,
          'drop_rogue': True,
          'hidden_size': 32,
          'lstm_layers': 2,
          'learning_rate': 0.01,
          'causal_attention': True,
          'max_encoder_length': 7,
          'attention_head_size': 2,
          'hidden_continuous_size': 16,
          'max_pred_length': 5
        },
        "lstm_model": {
          'max_encoder_length': 7,
          'max_pred_length': 5,
          'batch_size': 32,
          'drop_rogue': True,
          'n_features': 5,
          'n_hidden': 64,
          'n_layers': 2,
          'dropout': 0.2,

        }
    },
}

sweep_CFG = {
    "early_terminate": {
        "min_iter": 5,
        "type": "hyperband"
        },
    "method": "bayes",
    "metric": {
        "goal": "minimize",
        "name": "val_MAE"
        },
    "name": "TFT_sweep_1",
    "parameters": {
        "drop_rogue": {
            "value": True
            },
        "optimizer": {
            "value": "ranger"
            },
        "attention_head_size": {
            "values": [1,2,3,4]
            },
        "batch_size": {
            "values": [32,64,128,256]
            },
        "causal_attention": {
            "value": True
            },
        "dropout": {
            "values": [
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8
                ]
            },
        "epochs": {
            "value": 50
            },
        "hidden_continuous_size": {
            "values": [
                8,
                16,
                32,
                64,
                ]
            },
        "hidden_size": {
            "values": [
                16,
                32,
                64,
                128
                ]
            },
        "learning_rate": {
            "values": [0.001, 0.005, 0.01, 0.1]
            },
        "lstm_layers": {
            "values": [
                1,
                2,
                3,
                4
                ]
            },
        "max_encoder_length": {
            "values": [
                4,
                7,
                14,
                ]
            },
        },
}
