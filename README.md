# I-JEPA
```
ijepa/
├── configs/                  # Hyperparameters and experiment settings
├── src/                      # The core logic and neural networks
│   ├── datasets/             # Data loading and augmentation
│   ├── masks/                # The logic for generating context/target blocks
│   ├── models/               # The PyTorch neural network definitions
│   │   └── vision_transformer.py
│   └── utils/                # Logging, learning rate schedulers, EMA math
├── evals/                    # Code for testing the model after training
├── main.py                   # Single-GPU execution and debugging
└── main_distributed.py       # Multi-GPU (Slurm) training loop entry point
```
