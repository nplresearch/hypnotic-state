"""
Utility functions to perform an analysis starting with data, 
then embedding, them persistent homology.
"""

import numpy as np
import pandas as pd
from gtda.time_series import TakensEmbedding
from persim.persistent_entropy import persistent_entropy
from ripser import ripser
from scipy.optimize import root_scalar
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm

__all__ = [
    "trim",
    "downsample",
    "windowing",
    "temporal_embedding",
    "correlation_embedding",
    "persistent_homology",
    "getGreedyPerm",
    "getApproxSparseDM",
    "UDiPH",
    "sparse_ripser",
    "total_persistence",
    "betti_number",
    "slicing_loop",
    "persistent_homology_loop",
    "topological_features_loop",
]

# TODO
# filtration
# extract topological features
# def threshold_matrix(distance_matrix, threshold)
# generate null models

# ===============
# Data processing
# ===============


def trim(series, i_ini=None, i_end=None, max_length=None):
    """Trim a time series.

    Parameters
    ----------
    series : list
        A time series than can be uni- (n_t, ) or multi-variate (n_ch, n_t), in which case
        the time axis needs to be last.
    i_ini : int, optional
        The starting index for the trimming, by default None.
    i_end : int, optional
        The ending index for the trimming, by default None.
    max_length : int, optional
        The maximum length of the trimmed series, by default None.

    Returns
    -------
    list
        The trimmed time series.

    Note
    ----
    If max_length is not None, i_ini and i_end are ignored.
    Else, i_ini and i_end are used.

    Examples
    --------
    >>> trim([1, 2, 3, 4, 5], max_length=3)
    [1, 2, 3]

    >>> trim([1, 2, 3, 4, 5], i_ini=1, i_end=3)
    [2, 3]

    """

    if max_length and (i_ini or i_end):
        raise ValueError("Use either max_length or i_ini and i_end.")

    if not max_length and not i_ini and not i_end:
        raise ValueError("What the heck, I'm not doing anything to you data.")

    if max_length:
        length = min(len(series), max_length)
        return series[..., :length]

    if i_ini or i_end:
        return series[..., i_ini:i_end]


def downsample(series, skip):
    """
    Downsample the input `series` by a given factor.

    Parameters
    ----------
    series : list or array-like
        A time series than can be uni- (n_t, ) or multi-variate (n_ch, n_t), in which case
        the time axis needs to be last.
    skip : int
        The factor by which the data is to be downsampled.

    Returns
    -------
    list or array-like
        The downsampled data.

    Examples
    --------
    >>> downsample([1, 2, 3, 4, 5], 2)
    [1, 3, 5]
    """

    return series[..., ::skip]


def windowing(series, n_windows):
    """
    Divide the input `series` into `n_windows` equal parts.

    Parameters
    ----------
    series : list or array-like
        A time series than can be uni- (n_t, ) or multi-variate (n_ch, n_t), in which case
        the time axis needs to be last.
    n_windows : int
        The number of windows into which the data is to be divided.

    Returns
    -------
    list of lists or array-likes
        The divided windows.

    Examples
    --------
    >>> windowing([1, 2, 3, 4, 5], 2)
    [[1, 2], [3, 4]]
    """

    n_t = series.shape[-1]

    width = n_t // n_windows

    return [series[..., j * width : (j + 1) * width] for j in range(n_windows)]


# ===============
# Embeddings
# ===============


def temporal_embedding(series, delay, dimension):
    """
    Compute a temporal embedding of a time series.

    Parameters
    ----------
    series : array-like
        The input time series, either univariate (n_t,) or multivariate (n_ch, n_t).
    delay : int
        The time delay to be used for the temporal embedding.
    dimension : int
        The number of dimensions of the temporal embedding.

    Returns
    -------
    array-like
        The temporal embedding of the input time series.

    Examples
    --------
    >>> import numpy as np
    >>> series = np.array([1, 2, 3, 4, 5])
    >>> temporal_embedding(series, delay=1, dimension=2)
    array([[1., 2.],
           [2., 3.],
           [3., 4.],
           [4., 5.]])
    """

    if len(series.shape) == 1:
        multivariate = False
    if len(series.shape) == 2:
        multivariate = True
    else:
        raise ValueError("")

    TE = TakensEmbedding(time_delay=delay, dimension=dimension, flatten=multivariate)

    return TE.fit_transform(series[np.newaxis, :])[0]


def correlation_embedding(series, kind=None):
    """
    Compute a correlation embedding of a time series.

    Parameters
    ----------
    series : array-like
        The input time series.
    kind : str, optional
        The kind of correlation to be computed, by default None.

    Returns
    -------
    array-like
        The correlation embedding of the input time series.

    Examples
    --------
    >>> import numpy as np
    >>> series = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    >>> correlation_embedding(series)
    array([[0., 1.],
           [1., 0.]])
    """

    correlation_matrix = np.corrcoef(series)
    distance_matrix = 1 - correlation_matrix

    return distance_matrix


# ===================
# Persistent homology
# ===================


def persistent_homology(
    X, max_dimension, distance_matrix, sparse=True, epsilon=None, replace_inf=True
):
    """
    Compute persistent homology of a time series or a distance matrix.

    Parameters
    ----------
    X : array-like
        The input embedding. If from correlation, a (square) distance matrix.
        If from temporal embdding, a point cloud (2-dim array) or a pre-computed distance matrix between them.
    max_dimension : int
        The maximum dimension for which to compute the persistent homology.
    distance_matrix : bool, optional
        Whether the input `X` is a distance matrix, by default False. See `X` for details.
    sparse : bool, optional
        Whether to compute a sparse approximation of the persistence diagrams, by default True.
    eps: float
        Epsilon approximation constant for the sparse approximation. Ignored if sparse is False.
    replace_inf : bool, optional
        Whether to replace infinite persistence values with the maximum value of the distance matrix, by default True.

    Returns
    -------
    dict
        The persistence diagrams of the input time series or distance matrix.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    >>> persistent_homology(X, max_dimension=2, distance_matrix=False, sparse=True, replace_inf=True)
    {'dgms': [[(0, 1.0), (1, 1.0)], []], 'bottleneck': [0.0, inf], 'wasserstein': [0.0, inf], 'landscape': [0.0, inf], 'heat': [0.0, inf]}
    """

    if distance_matrix:
        if X.shape[0] != X.shape[1]:
            raise ValueError("X cannot be a distance matrix because it is not square.")

    if sparse and not epsilon:
        raise ValueError(
            "For the sparse approximation, epsilon needs to be a positive float (not None)."
        )

    if sparse:
        if distance_matrix:
            dist_mat = X
        else:
            dist_mat = pairwise_distances(X, metric="euclidean")

        lambdas = getGreedyPerm(dist_mat)
        DSparse = getApproxSparseDM(lambdas, epsilon, dist_mat)
        diagrams = ripser(DSparse, distance_matrix=True, maxdim=max_dimension)
        diameter = np.max(DSparse)

    else:
        diagrams = ripser(X, maxdim=max_dimension, distance_matrix=distance_matrix)

        if distance_matrix:
            dist_mat = X
            diameter = np.max(dist_mat)
        else:
            if replace_inf:  # no need to recompute it diameter is not needed
                dist_mat = pairwise_distances(X, metric="euclidean")
                diameter = np.max(dist_mat)

    if replace_inf:
        for dim_h in range(max_dimension + 1):
            for i, g in enumerate(diagrams["dgms"][dim_h]):
                if g[1] == np.inf:
                    diagrams["dgms"][dim_h][i][1] = diameter

    return diagrams


# The following 3 functions come from the Ripser website
# https://ripser.scikit-tda.org/en/latest/notebooks/Approximate%20Sparse%20Filtrations.html


def getGreedyPerm(D):
    """
    A Naive O(N^2) algorithm to do furthest points sampling

    Parameters
    ----------
    D : ndarray (N, N)
        An NxN distance matrix for points

    Return
    ------
    lamdas: list
        Insertion radii of all points
    """

    N = D.shape[0]
    # By default, takes the first point in the permutation to be the
    # first point in the point cloud, but could be random
    perm = np.zeros(N, dtype=np.int64)
    lambdas = np.zeros(N)
    ds = D[0, :]
    for i in range(1, N):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, D[idx, :])
    return lambdas[perm]


def getApproxSparseDM(lambdas, eps, D):
    """
    Purpose: To return the sparse edge list with the warped distances, sorted by weight

    Parameters
    ----------
    lambdas: list
        insertion radii for points
    eps: float
        epsilon approximation constant
    D: ndarray
        NxN distance matrix, okay to modify because last time it's used

    Return
    ------
    DSparse: scipy.sparse
        A sparse NxN matrix with the reweighted edges
    """
    N = D.shape[0]
    E0 = (1 + eps) / eps
    E1 = (1 + eps) ** 2 / eps

    # Create initial sparse list candidates (Lemma 6)
    # Search neighborhoods
    nBounds = ((eps**2 + 3 * eps + 2) / eps) * lambdas

    # Set all distances outside of search neighborhood to infinity
    D[D > nBounds[:, None]] = np.inf
    [I, J] = np.meshgrid(np.arange(N), np.arange(N))
    idx = I < J
    I = I[(D < np.inf) * (idx == 1)]
    J = J[(D < np.inf) * (idx == 1)]
    D = D[(D < np.inf) * (idx == 1)]

    # Prune sparse list and update warped edge lengths (Algorithm 3 pg. 14)
    minlam = np.minimum(lambdas[I], lambdas[J])
    maxlam = np.maximum(lambdas[I], lambdas[J])

    # Rule out edges between vertices whose balls stop growing before they touch
    # or where one of them would have been deleted.  M stores which of these
    # happens first
    M = np.minimum((E0 + E1) * minlam, E0 * (minlam + maxlam))

    t = np.arange(len(I))
    t = t[D <= M]
    (I, J, D) = (I[t], J[t], D[t])
    minlam = minlam[t]
    maxlam = maxlam[t]

    # Now figure out the metric of the edges that are actually added
    t = np.ones(len(I))

    # If cones haven't turned into cylinders, metric is unchanged
    t[D <= 2 * minlam * E0] = 0

    # Otherwise, if they meet before the M condition above, the metric is warped
    D[t == 1] = 2.0 * (D[t == 1] - minlam[t == 1] * E0)  # Multiply by 2 convention

    return coo_matrix((D, (I, J)), shape=(N, N)).tocsr()


def UDiPH(X, n_neighbors=15, distance_matrix=False):
    """
    Returns a distance matrix corresponding to a local uniform distribution of the points.
    Basically creates a new metric space where points are uniformly sampled in space.
    Think of UMAP but with a shortest path algo on top.

    Parameters
    ----------
    X: numpy array
        dataset (n_samples, n_features) or distance matrix (n_samples,n_samples)

    n_neighbors: int
        number of nearest neighbours considered when creating the proximity graph.
        Too many and topo features are dissolved, too few are artifacts are created.
        Fairly robust.

    distance_matrix: bool
        Whether X is a distance matrix or not.

    Returns
    -------
    x: np.array(n_samples,n_samples)
        distance matrix of new space.
    """
    if distance_matrix:
        neigh = NearestNeighbors(n_neighbors=n_neighbors, metric="precomputed").fit(X)
    else:
        neigh = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    D, idx = neigh.kneighbors()

    A = np.zeros([len(X), len(X)])
    for i in range(A.shape[0]):
        for j in range(n_neighbors):
            A[i, idx[i, j]] = j
    scaled = D / np.max(D, axis=1)[:, None]

    sigs = []
    for i in range(scaled.shape[0]):

        def f(sigma):
            return np.sum(np.exp(-scaled[i] / sigma)) - np.log2(n_neighbors)

        sol = root_scalar(f, x0=0.5, x1=0.07, method="secant")
        sigs.append(sol.root)

    D = np.exp(-(scaled / ((np.array(sigs))[:, None])))
    D[:, 0] = np.zeros(D.shape[0])
    for i in range(A.shape[0]):
        A[i, :] = D[i, A[i, :].astype(int)]

    A = A + np.eye(len(X))
    A = A + A.T - (A * A.T)
    A[A != 0] = np.abs(1 - A[A != 0])
    np.fill_diagonal(A, 0)

    x = shortest_path(A, directed=False)
    return x


def sparse_ripser(X, eps, max_dimension, replace_inf=True):
    """Utility function to compute persistent homology using a sparse approximation

    Parameters
    ----------
    X : array-like
        The input embedding. If from correlation, a (square) distance matrix.
        If from temporal embdding, a point cloud (2-dim array) or a pre-computed distance matrix between them.
    max_dimension : int
        The maximum dimension for which to compute the persistent homology.
    eps: float
        Epsilon approximation constant for the sparse approximation. Ignored if sparse is False.
    replace_inf : bool, optional
        Whether to replace infinite persistence values with the maximum value of the distance matrix, by default True.

    Returns
    -------
    dict
        The persistence diagrams of the input time series or distance matrix, in ripser format

    """

    dist_mat = pairwise_distances(X, metric="euclidean")
    lambdas = getGreedyPerm(dist_mat)
    DSparse = getApproxSparseDM(lambdas, eps, dist_mat)
    diagrams = ripser(DSparse, distance_matrix=True, maxdim=max_dimension)

    if replace_inf:
        diameter = np.max(dist_mat)
        for dim_h in range(max_dimension + 1):
            for i, g in enumerate(diagrams["dgms"][dim_h]):
                if g[1] == np.inf:
                    diagrams["dgms"][dim_h][i][1] = diameter

    return diagrams


def total_persistence(diagram_h):
    """Returns the total persistence of the persistence diagram

    Parameters
    ----------
    diagram_h : array
        Persistence diagram at dimension h

    Returns
    -------
    float
    """
    persistences = np.diff(diagram_h)
    return np.sum(persistences)


def betti_number(diagram_h):
    """Returns the total number of points in the persistence diagram

    Parameters
    ----------
    diagram_h : array
        Persistence diagram at dimension h

    Returns
    -------
    float
    """
    return len(diagram_h)


# =============================
# Wrapper to loop over series
# =============================


def slicing_loop(data_df, n_windows, skip=None):
    """
    Preprocesses time series data by cutting it into subwindows of equal length and then downsampling.

    Parameters
    ----------
    data_df : pandas.DataFrame
        A pandas DataFrame containing the time series data to be preprocessed, in
        a column called "series".
    n_windows : int
        The number of subwindows of equal length to cut each time series into.
    skip : int
        The downsampling rate. Each subwindow will be downsampled by this factor.

    Returns
    -------
    pandas.DataFrame
        A pandas DataFrame containing the preprocessed data. The DataFrame has one row per subwindow
        and the following columns: 'file' (the original file name), 'window' (the subwindow index),
        and 'series' (the preprocessed subwindow data).

    Examples
    --------
    >>> data_df = pd.DataFrame({'file': ['file1', 'file2'], 'series': [np.arange(6), np.arange(6, 12)]})
    >>> slicing_loop(data_df, n_windows=2, skip=2)
          file  window  series
    0   file1    0      [0, 2]
    1   file1    1      [3, 5]
    2   file2    0      [6, 8]
    3   file2    1      [9, 11]
    """

    data_cut = {}

    # time series pre-processing
    for series_idx in tqdm(data_df.index):
        series = data_df.loc[series_idx]["series"]

        # cut into subwindows of equal lenghts
        windows = windowing(series, n_windows)

        # downsample
        if skip:
            windows = [downsample(win, skip=skip) for win in windows]

        for j in range(n_windows):
            key_tmp = f"{series_idx}_{j}"  # temporary key, will be re-indexed later

            data_cut[key_tmp] = {}

            data_cut[key_tmp]["file"] = data_df.loc[series_idx]["file"]
            data_cut[key_tmp]["window"] = j
            data_cut[key_tmp]["series"] = windows[j]

    data_cut_df = pd.DataFrame.from_dict(data_cut, orient="index")

    return data_cut_df.reset_index(drop=True)


def persistent_homology_loop(
    data_df,
    max_dimension,
    embedding_kind,
    sparse=True,
    epsilon=None,
    replace_inf=True,
    **kwargs,
):
    """
    Compute persistent homology for each row in input `data_df`.

    Parameters
    ----------
    data_df : pandas DataFrame
        The input data, with each row corresponding to a recording. Must contain a column called "series"
        containing the time series.
    max_dimension : int
        The maximum dimension for which to compute the persistent homology.
    embedding_kind : {"temporal", "correlation"}
        Kind of embedding to perform
    sparse : bool, optional
        Whether to compute a sparse approximation of the persistence diagrams, by default True.
    eps: float
        Epsilon approximation constant for the sparse approximation. Ignored if sparse is False.
    replace_inf : bool, optional
        Whether to replace infinite persistence values with the maximum value of the distance matrix, by default True.
    **kwargs :
        Arguments `delay` and `dimension` to pass to `temporal_embedding()`

    Returns
    -------
    dict
        Persistence diagrams keyed by data_df index
    """

    diagrams_dict = dict()

    for series_idx in tqdm(data_df.index):
        # load series
        series = data_df.loc[series_idx]["series"]

        # perform embedding
        if embedding_kind == "temporal":
            X = temporal_embedding(series, **kwargs)
            distance_matrix = False
        elif embedding_kind == "correlation":
            X = correlation_embedding(series)
            distance_matrix = True

        # perform persistent homology
        diagrams = persistent_homology(
            X, max_dimension, distance_matrix, sparse, epsilon, replace_inf
        )

        diagrams_dict[series_idx] = diagrams

    return diagrams_dict


def topological_features_loop(diagrams_dict, max_dimension, scaling=None):
    """
    Computes topological features of persistence diagrams up to a given maximum dimension.

    Parameters
    ----------
    diagrams_dict : dict
        A dictionary where each key is a data file name, and the value is a dictionary containing the
        persistence diagrams (in ripser format) for different dimensions under the key "dgms".
    max_dimension : int
        The maximum dimension of the persistence diagrams to compute topological features for.
    scaling: {"minmax", "max", None}, optional
        If not None (default), scale values of each topological feature so that they are comparable.

    Returns
    -------
    dict
        A dictionary containing the computed topological features for each dimension and data file.
        The keys are the dimensions, and the values are pandas dataframes. The dataframes have data file names
        as index, and the topological features as columns.
    """

    topo_features = dict()

    for dim in range(max_dimension + 1):
        topo_features[dim] = {}

        for series_idx in diagrams_dict:
            topo_features[dim][series_idx] = {}

            diagram = diagrams_dict[series_idx]["dgms"][dim]

            topo_features[dim][series_idx]["tot_persistence"] = total_persistence(
                diagram
            )
            if total_persistence(diagram) > 0:
                topo_features[dim][series_idx][
                    "persistent_entropy"
                ] = persistent_entropy(diagram)[0]
            elif total_persistence(diagram) == 0:
                # added this because persistent_entropy does not handle 0
                topo_features[dim][series_idx]["persistent_entropy"] = 0
            else:
                raise ValueError("Total persistence should be non-negative")

            topo_features[dim][series_idx]["betti"] = betti_number(diagram)

        topo_features[dim] = pd.DataFrame.from_dict(topo_features[dim], orient="index")

        if scaling == "max":  # divide each feature by its max value
            topo_features[dim] = topo_features[dim] / topo_features[dim].max()
        elif scaling == "minmax":
            topo_features[dim] = pd.DataFrame(
                minmax_scale(topo_features[dim]),
                columns=topo_features[dim].columns,
                index=topo_features[dim].index,
            )

    return topo_features
