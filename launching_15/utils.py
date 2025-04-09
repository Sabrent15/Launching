

"""
    Perform bilinear interpolation, given grid vectors X, Y and a matrix V of values,
    plus query arrays x and y.
    
    The input arrays X and Y must be strictly increasing.

    Potentially replace the logic with python native packaages
"""
def lininterp2_vec_v2(X, Y, V, x, y):
    """
    
    Parameters
    ----------
    X : array-like, shape (n,)
        1D array of x coordinates (grid lines). Must be strictly increasing.
    Y : array-like, shape (m,)
        1D array of y coordinates (grid lines). Must be strictly increasing.
    V : 2D array, shape (n, m)
        2D array of function values defined on the grid of X and Y.
        The number of rows must equal len(X) and the number of columns must equal len(Y).
    x : array-like, shape (k,)
        Query points in the x direction.
    y : array-like, shape (k,)
        Query points in the y direction.
    
    Returns
    -------
    v : ndarray, shape (k,)
        Interpolated values at each (x, y) query.
    
    Raises
    ------
    ValueError
        If the sizes of X, Y, and V are inconsistent or if any x is outside the range of X.
    """
    # Convert inputs to 1D numpy arrays as needed
    X = np.atleast_1d(np.squeeze(np.array(X)))
    Y = np.atleast_1d(np.squeeze(np.array(Y)))
    x = np.atleast_1d(np.squeeze(np.array(x)))
    y = np.atleast_1d(np.squeeze(np.array(y)))
    
    # Verify that the dimensions of V match the lengths of X and Y
    if X.size != V.shape[0] or Y.size != V.shape[1]:
        raise ValueError(f"[len(X), len(Y)] does not match size(V). Got len(X)={X.size}, len(Y)={Y.size}, "
                         f"but V.shape={V.shape}")
    
    # For consistency with MATLAB behavior, ensure X and Y are column-like
    # (not strictly needed in Python, but we enforce 1D arrays)
    
    # --- Process x coordinate ---
    # Ensure x values are within bounds; MATLAB discretize would return NaN otherwise.
    if np.any(x < X[0]) or np.any(x > X[-1]) or np.any(x == X[-1]):
        raise ValueError(f"Input x values (min={x.min()}, max={x.max()}) are outside the range described by "
                         f"X[0]={X[0]} and X[-1]={X[-1]}.")
    
    # Determine the bin indices for x.
    # np.searchsorted with 'right' returns the first index where x would be inserted to maintain order.
    # Subtract 1 to get the index of the lower bound.
    pindexx = np.searchsorted(X, x, side='right') - 1
    indexx = pindexx + 1  # Upper index
    
    # Calculate the relative distance between the grid points
    X_lower = X[pindexx]
    X_upper = X[indexx]
    slopex = (x - X_lower) / (X_upper - X_lower)
    
    # --- Process y coordinate ---
    # For y, if out-of-bound values occur, issue a warning and assign y to the boundary values.
    pindexy = np.searchsorted(Y, y, side='right') - 1
    indexy = pindexy + 1
    
    # Find indices where y is out-of-bound. Use a mask for below and above range.
    below_mask = y < Y[0]
    above_mask = y > Y[-1]
    
    if np.any(below_mask) or np.any(above_mask):
        warnings.warn(f"Input y values (min={y.min()}, max={y.max()}) are outside the range described by "
                      f"Y[0]={Y[0]} and Y[-1]={Y[-1]}. Values below Y[0] are set to Y[0] and above Y[-1] are set to Y[-1].")
        # For values above the range:
        idx_above = np.where(above_mask)[0]
        pindexy[idx_above] = Y.size - 2  # lower index is second-last index
        indexy[idx_above] = Y.size - 1   # upper index is last index
        y[idx_above] = Y[-1]
        # For values below the range:
        idx_below = np.where(below_mask)[0]
        pindexy[idx_below] = 0
        indexy[idx_below] = 1
        y[idx_below] = Y[0]
    
    Y_lower = Y[pindexy]
    Y_upper = Y[indexy]
    slopey = (y - Y_lower) / (Y_upper - Y_lower)
    
    # --- Bilinear interpolation ---
    # For a query point, the bilinear interpolation is:
    # v = V(x_low, y_low)*(1-slopex)*(1-slopey) +
    #     V(x_high, y_low)*slopex*(1-slopey) +
    #     V(x_low, y_high)*(1-slopex)*slopey +
    #     V(x_high, y_high)*slopex*slopey
    v = (V[pindexx, pindexy] * (1 - slopex) * (1 - slopey) +
         V[indexx, pindexy]   * slopex       * (1 - slopey) +
         V[pindexx, indexy]   * (1 - slopex) * slopey +
         V[indexx, indexy]    * slopex       * slopey)
    
    return v
