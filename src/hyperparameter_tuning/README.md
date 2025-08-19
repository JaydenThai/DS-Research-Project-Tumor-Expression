# Hyperparameter Tuning System

This directory contains a comprehensive hyperparameter optimization system with a focus on gradient descent parameters.

## Features

- **Comprehensive Configuration Management**: Load/save hyperparameter configurations
- **Multiple Optimization Strategies**: Random search, grid search, Bayesian optimization
- **Extensive Gradient Descent Support**: Full parameter control for all major optimizers
- **Advanced Scheduler Support**: Plateau, cosine, step, exponential learning rate schedules
- **Gradient Analysis**: Clipping, norm tracking, and gradient-based optimization
- **Easy Integration**: Simple APIs for using optimized parameters in training

## Directory Structure

```
hyperparameter_tuning/
├── __init__.py                     # Module initialization
├── config.py                       # Configuration management
├── search_spaces.py               # Search space definitions
├── tuner.py                       # Main optimization implementation
├── migrate_config.py              # Legacy config migration
├── gradient_descent_optimization.py # Gradient descent focused optimization
├── results/                       # Optimization results storage
└── *.json                        # Saved configurations
```

## Quick Start

### Using Best Hyperparameters

```python
from src.hyperparameter_tuning import load_best_config
from src.utils.hyperparameter_training import quick_train_with_best_params

# Train with automatically loaded best parameters
results = quick_train_with_best_params(
    ModelClass, train_dataset, val_dataset, device
)
```

### Running Optimization

```python
from src.hyperparameter_tuning import HyperparameterTuner, SearchSpaceType

tuner = HyperparameterTuner(train_dataset, val_dataset, device_manager, ModelClass)

# Run random search with gradient descent focus
results = tuner.random_search(
    n_trials=50,
    search_space_type=SearchSpaceType.GRADIENT_DESCENT_FOCUSED
)

# Save best configuration
best_config = tuner.optimize_and_save_best()
```

### Creating Models with Best Parameters

```python
from src.models.cnn.model import PromoterCNN

# Automatically uses best hyperparameters
model = PromoterCNN.from_best_config()
```

## Supported Optimizers & Parameters

### Adam / AdamW

- `learning_rate`, `weight_decay`
- `beta1`, `beta2`, `eps`, `amsgrad`

### SGD

- `learning_rate`, `weight_decay`
- `momentum`, `nesterov`, `dampening`

### RMSprop

- `learning_rate`, `weight_decay`
- `alpha`, `centered`, `momentum`

## Search Spaces

- **BASIC**: Simple parameter ranges for quick optimization
- **COMPREHENSIVE**: Full parameter exploration
- **GRADIENT_DESCENT_FOCUSED**: Extensive gradient descent parameter tuning
- **ARCHITECTURE_FOCUSED**: Model architecture optimization
- **QUICK**: Fast optimization with reduced parameter space

## Running Gradient Descent Optimization

```bash
cd src/hyperparameter_tuning
python gradient_descent_optimization.py
```

This will run comprehensive gradient descent parameter optimization including:

- Optimizer comparison (Adam, AdamW, SGD, RMSprop)
- Learning rate schedule analysis
- Gradient clipping studies
- Extensive parameter sweeps

## Configuration Files

Best hyperparameters are automatically saved as:

- `promoter_cnn_best_config.json` - PromoterCNN optimal parameters
- Additional model configurations as needed

## Integration with Training

The system integrates seamlessly with training code:

```python
from src.utils.hyperparameter_training import HyperparameterTrainer

trainer = HyperparameterTrainer('promoter_cnn')
model = trainer.create_model(PromoterCNN)
results = trainer.train_model(model, train_dataset, val_dataset, device)
```

## Notebooks

- `enhanced_hyperparameter_tuning.ipynb` - Interactive optimization demonstrations
- `hyperparameter_optimization.ipynb` - Legacy optimization (for reference)

## Legacy Support

The system automatically migrates legacy hyperparameter files from `notebooks/experiments/best_hyperparameters.json` to the new format.
