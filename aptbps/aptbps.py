from functools import partial
import multiprocessing
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from scipy import stats
from sklearn.neighbors import KernelDensity
from KDEpy import FFTKDE, utils

# From https://github.com/sergeyprokudin/bps
def normalize(x, known_scalers=False, x_mean=None, x_max=None, max_rescale=True, return_scalers=False, verbose=False):
    """Normalize point clouds to fit a unit sphere

    Parameters
    ----------
    x : [n_clouds, n_points, n_dims]
        Input point clouds
    known_scalers: bool
        if True, use provided x_mean and x_max scalers for normalization
    max_rescale: bool
        if False, normalization will only include shifting by the mean value
    x_mean : numpy array [n_clouds, n_dims]
        if provided, mean value for every cloud used for normalization
    x_max : None or numpy array [n_clouds, 1]
        if provided, max norm for every cloud used for normalization
    return_scalers: bool
        whether to return point cloud scalers (needed for denormalisation)
    verbose: bool
        whether to print progress
    Returns
    -------
    x_norm : numpy array [n_clouds, n_points, n_dims]
        Normalized point clouds
    x_mean : numpy array [n_clouds, n_dims]
        Mean value of every cloud
    x_max : numpy array [n_clouds, 1]
        Max norm of every cloud
    """

    def _normalize_cloud(x, x_mean=None, x_max=None, max_rescale=True):
        """normalize single cloud"""

        if x_mean is None:
            x_mean = np.mean(x, axis=0)

        x_norm = np.copy(x - x_mean)

        # note: max norm could be not robust to outliers!
        if x_max is None:
            if max_rescale:
                x_max = np.max(np.sqrt(np.sum(np.square(x), axis=1)))
            else:
                x_max = 1.0
        x_norm = x_norm / x_max

        return x_norm, x_mean, x_max

    n_clouds, n_points, n_dims = x.shape

    x_norm = np.zeros([n_clouds, n_points, n_dims])

    if known_scalers is False:
        x_mean = np.zeros([n_clouds, n_dims])
        x_max = np.zeros([n_clouds, 1])

    fid_lst = range(0, n_clouds)

    if verbose:
        fid_lst = tqdm(fid_lst)

    for pid in fid_lst:
        if known_scalers is False:
            x_norm[pid], x_mean[pid], x_max[pid] = _normalize_cloud(x[pid])
        else:
            x_norm[pid], _, _ = _normalize_cloud(x[pid], x_mean[pid], x_max[pid], max_rescale=max_rescale)

    if return_scalers:
        return x_norm, x_mean, x_max
    else:
        return x_norm

# From https://github.com/sergeyprokudin/bps
def denormalize(x_norm, x_mean, x_max):
    """Denormalize point clouds

    Parameters
    ----------
    x : [n_clouds, n_points, n_dims]
        Input point clouds
    rescale: bool
        if False, normalization will only include shifting by the mean value
    x_mean : numpy array [n_clouds, n_dims]
        if provided, mean value for every cloud used for normalization
    x_max : None or numpy array [n_clouds, 1]
        if provided, max norm for every cloud used for normalization

    Returns
    -------
    x_norm : numpy array [n_clouds, n_points, n_dims]
        Normalized point clouds
    x_mean : numpy array [n_clouds, n_dims]
        Mean value of every cloud
    x_max : numpy array [n_clouds, 1]
        Max norm of every cloud
    """

    def _denormalize_cloud(x_norm, x_mean, x_max):
        """denormalize single cloud"""

        x_denorm = x_norm * x_max + x_mean

        return x_denorm

    x_denorm = np.zeros(x_norm.shape)

    for pid in range(0, len(x_norm)):
        x_denorm[pid] = _denormalize_cloud(x_norm[pid], x_mean[pid], x_max[pid])

    return x_denorm

# From https://github.com/sergeyprokudin/bps
def generate_random_basis(n_points=1000, n_dims=3, radius=1.0, random_seed=13):
    """Sample uniformly from d-dimensional unit ball

    The code is inspired by this small note:
    https://blogs.sas.com/content/iml/2016/04/06/generate-points-uniformly-in-ball.html

    Parameters
    ----------
    n_points : int
        number of samples
    n_dims : int
        number of dimensions
    radius: float
        ball radius
    random_seed: int
        random seed for basis point selection
    Returns
    -------
    x : numpy array
        points sampled from d-ball
    """
    np.random.seed(random_seed)
    # sample point from d-sphere
    x = np.random.normal(size=[n_points, n_dims])
    x_norms = np.sqrt(np.sum(np.square(x), axis=1)).reshape([-1, 1])
    x_unit = x / x_norms

    # now sample radiuses uniformly
    r = np.random.uniform(size=[n_points, 1])
    u = np.power(r, 1.0 / n_dims)
    x = radius * x_unit * u
    np.random.seed(None)

    return x

# From https://github.com/sergeyprokudin/bps
def generate_grid_basis(grid_size=32, n_dims=3, minv=-1.0, maxv=1.0):
    """ Generate d-dimensional grid BPS basis

    Parameters
    ----------
    grid_size: int
        number of elements in each grid axis
    minv: float
        minimum element of the grid
    maxv
        maximum element of the grid

    Returns
    -------
    basis: numpy array [grid_size**n_dims, n_dims]Taddl
        n-d grid points
    """

    linspaces = [np.linspace(minv, maxv, num=grid_size) for d in range(0, n_dims)]
    coords = np.meshgrid(*linspaces)
    basis = np.concatenate([coords[i].reshape([-1, 1]) for i in range(0, n_dims)], axis=1)

    return basis

# Modified from https://github.com/sergeyprokudin/bps
def encode(x, bps_arrangement='random', n_bps_points=512, radius=1.5, bps_cell_type='dists', return_idx=False, nn_algo="kd_tree",
           leaf_size=16, verbose=1, random_seed=13, x_features=None, custom_basis=None, n_jobs=-1, normalize=False):
    """Converts point clouds to basis point set (BPS) representation, multi-processing version

    Parameters
    ----------
    x: numpy array [n_clouds, n_points, n_dims]
        batch of point clouds to be converted
    bps_arrangement: str
        supported BPS arrangements: "random", "grid", "custom"
    n_bps_points: int
        number of basis points
    radius: float
        radius for BPS sampling area
    bps_cell_type: str
        type of information stored in every BPS cell. Supported:
            'dists': Euclidean distance to the nearest point in cloud
            'deltas': delta vector from basis point to the nearest point
            'closest': closest point itself
            'features': return features of the closest point supplied by x_features.
                        e.g. RGB values of points, surface normals, etc.
    return_idx: boolean
        whether to return the indexes of the sampled points.
    nn_algo: str
        what algorithm to use to compute nearest neighbor searches. Supported (from sklearn docs):
            'ball_tree': will use :class:`BallTree`
            'kd_tree': will use :class:`KDTree`
            'brute': will use a brute-force search.
            'auto': will attempt to decide the most appropriate algorithm
                    based on the values passed to :meth:`fit` method.
    normalize: boolean
        normalize data before encoding (default=False)
    leaf_size: int
        leaf size of the decision tree.
    verbose: boolean
        whether to show conversion progress
    x_features: numpy array [n_clouds, n_points, n_features]
        point features that will be stored in BPS cells if return_values=='features'
    custom_basis: numpy array [n_basis_points, n_dims]
        custom basis to use
    n_jobs: int
        number of parallel jobs used for encoding. If -1, use all available CPUs

    Returns
    -------
    x_bps: [n_clouds, n_points, n_bps_features]
        point clouds converted to BPS representation.
    idx_bps: [n_clouds, n_points] (only returned if return_idx=True)
        indexes of the points sampled by the BPS encoding.
    """

    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    if n_jobs == 1:

        if normalize:
            x = normalize(x)

        n_clouds, n_points, n_dims = x.shape

        if bps_arrangement == 'random':
            basis_set = generate_random_basis(n_bps_points, n_dims=n_dims, radius=radius, random_seed=random_seed)
        elif bps_arrangement == 'grid':
            # in case of a grid basis, we need to find the nearest possible grid size
            grid_size = int(np.round(np.power(n_bps_points, 1 / n_dims)))
            basis_set = generate_grid_basis(grid_size=grid_size, minv=-radius, maxv=radius)
        elif bps_arrangement == 'custom':
            # in case of a grid basis, we need to find the nearest possible grid size
            if custom_basis is not None:
                basis_set = custom_basis
            else:
                raise ValueError("Custom BPS arrangement selected, but no custom_basis provided.")
        else:
            raise ValueError("Invalid basis type. Supported types: \'random\', \'grid\', \'custom\'")

        n_bps_points = basis_set.shape[0]

        if bps_cell_type == 'dists':
            x_bps = np.zeros([n_clouds, n_bps_points])
        elif bps_cell_type == 'deltas':
            x_bps = np.zeros([n_clouds, n_bps_points, n_dims])
        elif bps_cell_type == 'closest':
            x_bps = np.zeros([n_clouds, n_bps_points, n_dims])
        elif bps_cell_type == 'features':
            n_features = x_features.shape[2]
            x_bps = np.zeros([n_clouds, n_bps_points, n_features])
        else:
            raise ValueError("Invalid cell type. Supported types: \'dists\', \'deltas\', \'closest\', \'features\'")
        fid_lst = range(0, x.shape[0])

        idx_bps = np.zeros([n_clouds, n_bps_points])
        
        if verbose:
            fid_lst = tqdm(fid_lst)

        for fid in fid_lst:
            nbrs = NearestNeighbors(n_neighbors=1, leaf_size=leaf_size, algorithm=nn_algo).fit(x[fid])
            fid_dist, npts_ix = nbrs.kneighbors(basis_set)
            if bps_cell_type == 'dists':
                x_bps[fid] = fid_dist.squeeze()
            elif bps_cell_type == 'deltas':
                x_bps[fid] = x[fid][npts_ix].squeeze() - basis_set
            elif bps_cell_type == 'closest':
                x_bps[fid] = x[fid][npts_ix].squeeze()
            elif bps_cell_type == 'features':
                x_bps[fid] = x_features[fid][npts_ix].squeeze()
            # Store index of sampled point in idx_bps
            # This is done even when return_idx=False;
            idx_bps[fid] = npts_ix.squeeze()

        if return_idx:
            return x_bps, idx_bps
        else:
            return x_bps

    else:
        
        if verbose:
            print("using %d available CPUs for BPS encoding.." % n_jobs)
        
        bps_encode_func = partial(encode, bps_arrangement=bps_arrangement, n_bps_points=n_bps_points, radius=radius,
                                  bps_cell_type=bps_cell_type, verbose=verbose, leaf_size=leaf_size, random_seed=random_seed, return_idx=return_idx,
                                  x_features=x_features, custom_basis=custom_basis, n_jobs=1, nn_algo=nn_algo, normalize=normalize)

        pool = multiprocessing.Pool(n_jobs)
        x_chunks = np.array_split(x, n_jobs)

        if return_idx:
            x_bps, idx_bps = zip(*pool.map(bps_encode_func, x_chunks))
            x_bps = np.concatenate(x_bps, 0)
            idx_bps = np.concatenate(idx_bps, 0)
            idx_bps = idx_bps.astype('uint16')
        else:
            x_bps = np.concatenate(pool.map(bps_encode_func, x_chunks), 0)
        
        pool.close()

        if return_idx:
            return x_bps, idx_bps
        else:
            return x_bps
        
def adaptive_encode(x, bps_arrangement='random', kde='gaussian', n_bps_points=512, n_parts=2, radius=1.5, bps_cell_type='dists',
           verbose=1, random_seed=13, x_features=None, custom_basis=None, n_jobs=-1, partition='triangle'):
    """Returns an APTBPS encoded cloud

    Parameters
    ----------
    x: numpy array [n_clouds, n_points, n_dims]
        batch of point clouds to be converted
    bps_arrangement: str
        supported BPS arrangements: "random", "grid", "custom"
    n_bps_points: int
        number of basis points
    radius: float
        radius for BPS sampling area
    kde: str
        type of kde used.
            'gaussian': Gaussian KDE
            'fft': Gaussian FFT KDE
    partition: str
        how to partition input.
            'comp': integer composition
            'triangle': triangle numbers
    bps_cell_type: str
        type of information stored in every BPS cell. Supported:
            'dists': Euclidean distance to the nearest point in cloud
            'deltas': delta vector from basis point to the nearest point
            'closest': closest point itself
            'features': return features of the closest point supplied by x_features.
                        e.g. RGB values of points, surface normals, etc.
    verbose: boolean
        whether to show conversion progress
    x_features: numpy array [n_clouds, n_points, n_features]
        point features that will be stored in BPS cells if return_values=='features'
    custom_basis: numpy array [n_basis_points, n_dims]
        custom basis to use
    n_jobs: int
        number of parallel jobs used for encoding. If -1, use all available CPUs

    Returns
    -------
    x_bps: [n_clouds, n_points, n_bps_features]
        point clouds converted to BPS representation.
    """

    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    if n_jobs == 1:

        if n_parts < 1 or not isinstance(n_parts, int):
            print("Error: number of partitions must be a positive integer bigger than or equal to 1")
            return
        # If there's only one partition, then just run the regular encode function 
        elif n_parts == 1:
            return encode(x=x, bps_arrangement=bps_arrangement, n_bps_points=n_bps_points,
                          radius=radius, bps_cell_type=bps_cell_type, verbose=verbose, random_seed=random_seed,
                          x_features=x_features, custom_basis=custom_basis, n_jobs=1)

        # Triangle number formula for finding the partition size
        part_size = (2*n_bps_points)//(n_parts*(n_parts+1))

        if part_size == 0:
            print("Error: please reduce number of partitions")
            return

        n_clouds, n_points, n_dims = x.shape

        if kde == 'fft':
            fft_grid = int(np.round(np.power(n_points, 1 / n_dims)))

        if bps_arrangement == 'random':
            basis_set = generate_random_basis(n_bps_points, n_dims=n_dims, radius=radius, random_seed=random_seed)
        elif bps_arrangement == 'grid':
            # in case of a grid basis, we need to find the nearest possible grid size
            grid_size = int(np.round(np.power(n_bps_points, 1 / n_dims)))
            basis_set = generate_grid_basis(grid_size=grid_size, minv=-radius, maxv=radius)
        elif bps_arrangement == 'custom':
            # in case of a grid basis, we need to find the nearest possible grid size
            if custom_basis is not None:
                basis_set = custom_basis
            else:
                raise ValueError("Custom BPS arrangement selected, but no custom_basis provided.")
        else:
            raise ValueError("Invalid basis type. Supported types: \'random\', \'grid\', \'custom\'")

        n_bps_points = basis_set.shape[0]

        if bps_cell_type == 'dists':
            x_bps = np.zeros([n_clouds, n_bps_points])
        elif bps_cell_type == 'deltas':
            x_bps = np.zeros([n_clouds, n_bps_points, n_dims])
        elif bps_cell_type == 'closest':
            x_bps = np.zeros([n_clouds, n_bps_points, n_dims])
        elif bps_cell_type == 'features':
            n_features = x_features.shape[2]
            x_bps = np.zeros([n_clouds, n_bps_points, n_features])
        else:
            raise ValueError("Invalid cell type. Supported types: \'dists\', \'deltas\', \'closest\', \'features\'")
        fid_lst = range(0, n_clouds)

        if verbose:
            fid_lst = tqdm(fid_lst)

        for fid in fid_lst:
            
            if kde == 'gaussian':
                try:
                    kde = stats.gaussian_kde(x[fid].T)
                except np.linalg.LinAlgError:
                    # If input matrix is singular,
                    # FFTKDE can be computed instead
                    # Skips this sample
                    continue
                input_density = kde(x[fid].T)

                # PARTITIONING STAGE
                input_density = input_density.reshape(-1, 1)

                input_density_cat = np.concatenate([x[fid], input_density], axis=1)
            
            elif kde == 'fft':
                coords, input_density = FFTKDE(kernel="gaussian", bw=1E-4).fit(x[fid]).evaluate(fft_grid+1)
                input_density = input_density.reshape(-1, 1)
                input_density_cat = np.concatenate([coords, input_density], axis=1)

            num_points = input_density_cat.shape[0]

            part_idxs = []

            for i in range(1, n_parts):
                part_idxs.append(i*num_points//n_parts)

            # partition in n_parts
            input_density_cat = input_density_cat[np.argpartition(input_density_cat[:, 3], part_idxs)]

            bps_parts = []
            fid_dists = []

            bps_deltas = []
            
            # Add these points to the basis points for the first iteration;
            # this variable will be set to 0 afer the first iteration.
            # This makes sure that the encoded cloud will be of size n_bps_points.
            extra_points = n_bps_points % part_size

            tot_points = 0

            for i in range (0, n_parts):
                tot_points = tot_points + (n_parts-i)*part_size

            tot_points = tot_points + extra_points
            basis_start_idx = 0

            if tot_points is not n_bps_points:
                extra_points = extra_points + (n_bps_points - tot_points)

            # ENCODING STAGE

            for i in range(0, n_parts):
                start_idx = (num_points//n_parts)*i
                end_idx = (num_points//n_parts) + start_idx # we lose n_parts-1 points at the end, but it's okay cause they're high density points

                # For current iteration
                n_curr_basis_points = (n_parts-i)*part_size + extra_points
                
                curr_basis_points = basis_set[basis_start_idx:basis_start_idx+n_curr_basis_points]

                basis_start_idx = n_curr_basis_points
                
                # set up tree for nearest neighbors
                nbrs = NearestNeighbors(n_neighbors=1, leaf_size=16, algorithm="kd_tree").fit(input_density_cat[start_idx:end_idx, 0:3])

                fid_dist, npts_ix = nbrs.kneighbors(curr_basis_points, n_neighbors=1, return_distance=True)

                # Need to sum start_idx+npts_ix, 
                # because when you fit start_idx:end_idx it indexes the current points from 0
                bps_parts.append(input_density_cat[start_idx+npts_ix, 0:3])
                fid_dists.append(fid_dist)
                
                if bps_cell_type == 'deltas':
                    bps_delta = bps_parts[i].squeeze() - curr_basis_points
                    bps_delta = bps_delta.reshape(-1, 3)
                    bps_deltas.append(bps_delta)

                extra_points = 0


            bps_parts_concat = np.vstack(bps_parts)

            fid_dists_concat = np.vstack(fid_dists)

            bps_parts_concat = bps_parts_concat.reshape(-1, n_bps_points, 3)

            if bps_cell_type == 'dists':
                x_bps[fid] = fid_dists_concat.squeeze()
            elif bps_cell_type == 'deltas':
                bps_deltas_concat = np.vstack(bps_deltas)
                bps_deltas_concat = bps_deltas_concat.reshape(-1, n_dims)
                x_bps[fid] = bps_deltas_concat.squeeze()
            elif bps_cell_type == 'closest':
                x_bps[fid] = bps_parts_concat.squeeze()
            elif bps_cell_type == 'features':
                x_bps[fid] = x_features[fid][npts_ix].squeeze()

        return x_bps

    else:
        if verbose:
            print("using %d available CPUs for BPS encoding.." % n_jobs)

        bps_adaptive_encode_func = partial(adaptive_encode, bps_arrangement=bps_arrangement, n_bps_points=n_bps_points, 
                                  radius=radius, n_parts=n_parts, kde=kde,
                                  bps_cell_type=bps_cell_type, verbose=verbose, random_seed=random_seed,
                                  x_features=x_features, custom_basis=custom_basis, n_jobs=1)

        pool = multiprocessing.Pool(n_jobs)
        x_chunks = np.array_split(x, n_jobs)
        x_bps = np.concatenate(pool.map(bps_adaptive_encode_func, x_chunks), 0)
        pool.close()

        return x_bps

def partition_min_max(n, k, l, m):
    """Partitions an integer.
    Source: https://stackoverflow.com/a/43015372
    
    Arguments:
    n: integer to partition
    k: number of partitions to partition n into
    l: minimum number that a partition can be
    m: maximum number that a partition can be"""

    if k < 1:
        return
    if k == 1:
        if n <= m and n >= l:
            yield (n,)
        return
    for i in range(l, m+1):
        for result in partition_min_max(n-i, k-1, i, m):                
            yield result+(i,)