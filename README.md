# RNANO

Deep learning tool for RNA modification detection from Nanopore direct RNA sequencing data.

## Features

- Detects multiple RNA modification types (m6A, m1A, m5C, m7G, ac4C, Nm, pU)
- Works with different cell lines (HEK293t, Hela, HepG2, IM95, hESCs)
- Attention-based neural network architecture
- Comprehensive evaluation metrics

## Installation

```bash
git clone https://github.com/abhhba999/RNANO.git
cd RNANO
pip install -r requirements.txt
```

## Quick Start

### Training

```bash
python train.py --mode Train --mod_type m7G --cell_line IM95 --batch_size 256 --learning_rate 0.003
```

### Prediction

```bash
python predict.py --mod_type m7G --cell_line IM95
```

## Project Structure

- `model.py`: Neural network architecture
- `dataset.py`: Data loading and preprocessing
- `utils.py`: Utility functions
- `train.py`: Training script
- `predict.py`: Prediction script
- `example.py`: Example usage

## Dependencies

- PyTorch
- NumPy
- Pandas
- scikit-learn
- Other dependencies in requirements.txt

## Citation

If you use RNANO in your research, please cite:

```
@article{RNANO2025,
  title={},
  author={},
  journal={},
  year={2025}
}
```

## License

MIT
