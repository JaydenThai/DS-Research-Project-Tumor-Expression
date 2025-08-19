# DS-Research-Project-Tumor-Expression

This repository contains data exploration notebooks and deep learning models for promoter activity prediction.

## Repository structure

```
DS-Research-Project-Tumor-Expression/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                    # Original datasets
│   ├── processed/              # Cleaned/transformed data
│   └── outputs/                # Generated plots, reports
├── notebooks/
│   ├── exploratory/            # Early exploration (bimodality, data_exploration)
│   ├── data_processing/        # Data cleaning/preparation (ICE-T_ProDat)
│   └── experiments/            # Model experiments (promoter_cnn, baseline_simple, dnabert)
├── src/
│   ├── models/
│   │   ├── baseline/           # Simple baseline models
│   │   ├── cnn/                # CNN model implementation
│   │   ├── dnabert/            # DNABERT model implementation
│   │   └── regression/         # Regression models
│   └── utils/                  # Shared utilities (data, training, viz)
├── results/
│   ├── model_weights/          # Trained model weights (.pth files)
│   ├── plots/                  # Generated visualizations
│   └── analysis/               # Analysis results and reports
└── docs/
    └── notes/                  # Meeting notes and documentation
```

## Quickstart

These instructions assume macOS with Homebrew and Python 3.10+ available. For Apple Silicon (M1/M2/M3), PyTorch via pip works; for conda users, see the conda section.

### 1) Clone and enter the repo

```bash
git clone <your-repo-url>
cd DS-Research-Project-Tumor-Expression
```

### 2) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On macOS, if you use `conda` instead of `venv`:

```bash
conda create -n tumor-exp python=3.10 -y
conda activate tumor-exp
```

### 3) Install dependencies

Pip users:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Conda users (recommended channels for scientific stack):

```bash
conda install -c conda-forge numpy pandas scikit-learn matplotlib seaborn jupyterlab ipykernel -y
# Install PyTorch from pytorch channel
conda install -c pytorch pytorch torchvision -y
```

If you have an Apple Silicon Mac and want Metal acceleration for PyTorch, recent PyTorch wheels include MPS support by default when installed via pip. No extra steps needed.

### 4) Register the kernel (optional)

If you plan to use JupyterLab with this environment:

```bash
python -m ipykernel install --user --name tumor-exp --display-name "Python (tumor-exp)"
```

### 5) Open notebooks / run models

- Open JupyterLab and run notebooks:

```bash
jupyter lab
```

- Promoter CNN assets are in:
  - `notebooks/experiments/promoter_cnn.ipynb`: main notebook
  - `results/model_weights/best_promoter_cnn.pth`: trained weights
  - `results/plots/cnn_results.png`: example results

### 6) Data locations

- Raw data: `data/raw/`
- Processed data: `data/processed/`
- Results and plots: `results/`

If paths in notebooks need adjusting, search for hard-coded paths and update accordingly.

## Reproducing the CNN results

1. Ensure dependencies are installed (see step 3).
2. Launch JupyterLab and open `notebooks/experiments/promoter_cnn.ipynb`.
3. The notebook expects data at `data/processed/` which should be populated after reorganization.
4. To use the provided weights, load from `results/model_weights/best_promoter_cnn.pth`.

## Project-specific requirements

- The model-specific requirements originally in `Main Work/models/requirements.txt` have been consolidated into the top-level `requirements.txt`.
- If future models have unique dependencies, add a `requirements.txt` inside that model's subfolder.

## Troubleshooting

- If `pip install torch` fails on macOS with SSL errors, try:

```bash
pip install --upgrade certifi
```

- If you use system Python on macOS and encounter permission errors, prefer using a virtual environment as shown above.

- If using conda and you need CUDA (Linux/NVIDIA), install from the official selector at `https://pytorch.org/get-started/locally/`.

## License

Specify your license here.
