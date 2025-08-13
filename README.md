# DS-Research-Project-Tumor-Expression

This repository contains data exploration notebooks and deep learning models for promoter activity prediction.

## Repository structure

- `Main Work/`
  - `Data/`: input and processed dataset files and plots
  - `models/`
    - `promoter_cnn/`: CNN model notebook, weights, results image, and requirements specific to the model
  - `Prelim/`: early exploratory notebooks

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

- Promoter CNN assets are in `Main Work/models/promoter_cnn/`:
  - `promoter_cnn.ipynb`: main notebook
  - `best_promoter_cnn.pth`: trained weights
  - `cnn_results.png`: example results

### 6) Data locations

- Raw/processed data: `Main Work/Data/` and `Main Work/Processed-Data/`

If paths in notebooks need adjusting, search for hard-coded paths and update accordingly.

## Reproducing the CNN results

1. Ensure dependencies are installed (see step 3).
2. Launch JupyterLab and open `Main Work/models/promoter_cnn/promoter_cnn.ipynb`.
3. If the notebook expects data at `Main Work/Data/` or `Main Work/Processed-Data/`, keep the repo layout intact.
4. To use the provided weights, load `best_promoter_cnn.pth` in the notebook cells that define `torch.load(...)`.

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
