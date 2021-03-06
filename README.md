# APTBPS

- Visit [aptbps-code](https://github.com/rdimaio/aptbps-code) for training and graphing code

![](https://github.com/rdimaio/aptbps-code/blob/master/graphing/animation/aptbps-stills.gif)

# Installation
## Requirements
- Python 3.x
- scikit-learn
- numpy
- tqdm
- KDEpy
- scipy

## `pip` installation
```bash
$ pip3 install git+https://github.com/rdimaio/aptbps
```

## From source

Or, if you prefer, you can install it from source:
```bash
# Clone the repository
$ git clone https://github.com/rdimaio/aptbps

# Go into the aptbps folder
$ cd aptbps

# Install aptbps
$ python setup.py install
```

# Usage
```python
import aptbps
from bps import bps

# x is a batch of point clouds in the shape [n_clouds, n_points, n_dims]

# Original bps implementation
x_bps = aptbps.encode(x, bps_cel) 

# 2-partition triangle-gkde aptbps encoding
x_aptbps = aptbps.adaptive_encode(x, n_parts=2, bps_cell_type='dists')

# 8-partition comp-fftkde aptbps encoding
x_aptbps_fftkde = aptbps.adaptive_encode(x, n_parts=8, kde='fft', partitioning='comp', bps_cell_type='dists')

```

# Resources
- [Pre-trained models](https://drive.google.com/open?id=198OyZhesh8bAljNCLjvBB9_TMWJurnC1)
- [Pre-encoded data](https://drive.google.com/open?id=1PN-kQVoee8pS-56wpHLUuWMR0aoFwLVU)

# Contributing
Pull requests are welcome! If you would like to include/remove/change a major feature, please open an issue first.

# Acknowledgement
[BPS repository](https://github.com/sergeyprokudin/bps)

# License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
