# Machine Unlearning in Deep Vision Models

This repository hosts the implementation and experimental framework for the study titled:
**"Understanding Machine Unlearning in Image Models: Layer Sensitivity, Forget-Set Scaling, Cross-Architecture Difficulty, Collateral Damage, and Cost–Quality Tradeoffs."**

The project investigates the behavior of machine unlearning techniques in deep image classification models, evaluating how different architectures (CNNs and Vision Transformers) forget information when specific subsets of training data are removed. The study includes gradient-ascent scrubbing, SISA-style partial retraining, and detailed evaluation metrics to quantify forgetting success, retention performance, embedding drift, runtime cost, and collateral effects.

---

## Repository Structure
```
deepvision-unlearning/
├── data/                   # Dataset utilities and loaders
├── models/                 # Baseline model architectures (CNNs and ViTs)
├── unlearning_methods/     # Gradient Ascent and SISA unlearning implementations
├── forget_sets/            # Forget-set creation and configuration
├── evaluation/             # Metrics: forgetting, retention, drift, confusion, runtime, MIA
├── experiments/            # Training, unlearning, evaluation experiment scripts
├── utils/                  # Trainer utilities and helper functions
├── configs/                # Class mappings and configuration files
├── logs/                   # Experiment logs and results
├── README.md
└── requirements.txt
```

---

## Requirements and Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/sri-sruthi/deepvision-unlearning.git
cd deepvision-unlearning
pip install -r requirements.txt
```

Run with GPU support (Google Colab or local CUDA recommended).

## Dataset

CIFAR-10 and CIFAR-100 are automatically downloaded when loaders are initialized. No manual dataset download or upload is required.

## Baseline Model Training

To train ResNet-18 on CIFAR-10:
```bash
python experiments/train.py --model resnet18 --dataset cifar10
```

Supported model options:
```
resnet18
mobilenetv2
vit_tiny
mobilevit
```

Supported datasets:
```
cifar10
cifar100
```

The best model checkpoint is automatically saved.

## Running Unlearning Methods

### Full Class Forgetting
```bash
python experiments/run_unlearning.py --method ga --model resnet18 --dataset cifar10 --forget-type class --forget-class airplane
```

### Random Sample Forgetting
```bash
python experiments/run_unlearning.py --method sisa --model resnet18 --dataset cifar10 --forget-type sample --forget-count 500
```

Unlearned model checkpoints are stored in `./models/`.

## Evaluation
```bash
python experiments/run_evaluation.py --model-path ./models/resnet18_cifar10_ga_class.pth
```

Outputs include:

* Forgetting Effectiveness (lower indicates successful forgetting)
* Retention Accuracy (higher indicates minimal loss on retained samples)

Advanced metrics available:

* Confusion matrix drift
* Feature embedding drift per layer
* Runtime comparison
* Membership inference attack success probability

## Research Objectives

This repository enables experiments evaluating:

1. Layer-wise sensitivity to unlearning
2. Forget-set scaling behavior
3. Cross-architecture unlearning difficulty (CNNs vs ViTs)
4. Collateral damage on non-forgotten classes
5. Cost–quality tradeoff between forgetting completeness and utility preservation

## Contact

Maintained by: Krishna Midula K and Sri Sruthi M N  
Institution: Amrita Vishwa Vidyapeetham, Coimbatore
