# APTBPS - Density-Adaptive Distance Encoding for Machine Learning on Point Clouds

<div align="center">
&nbsp;&nbsp;

  <a href="#key-features">Key Features</a> •
  <a href="#supported-formats">Supported Formats</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#related-projects">Related projects</a> •
  <a href="#contributing">Contributing</a> •
  <a href="#license">MIT License</a>
</div>

# Key features
- Visit [aptbps-code](https://github.com/rdimaio/aptbps-code) for training and graphing code

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

# Contributing
Pull requests are welcome! If you would like to include/remove/change a major feature, please open an issue first.

# License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
