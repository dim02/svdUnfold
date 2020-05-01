# svdUnfold

**svdUnfold** is a python implementation of singular value decomposition data unfolding based on [SVD Approach to Data Unfolding](https://www.sciencedirect.com/science/article/pii/0168900295014780).

----

### Using svdUnfold

```python
import svdunfold
unfold = svdunfold.SVDunfold(x_ini, b, A, B)
```

x_ini is a numpy histogram with the initial generated distribution, representing your expectation for the true underlying distribution. A is a numpy array of the response matrix describing the detector effect, e.g. smearing and efficiency. b is a numpy histogram containing the true measured data distribution and B is a numpy array of the covariance matrix of the measurement.


```python
unfold.transform_system()
d = unfold.get_abs_d()
```

transform_system() prepares the system for unfolding, calculating the decomposition coefficients d. After choosing an appropriate value for the regularization parameter k, simply do:

```python
unfold.unfold(k)
```

The unfolded distribution and the corresponding covariance matrix can be retrieved with:

```python
x_unfolded = unfold.get_unfolded_distribution()
X_covariance = unfold.get_unfolded_cov_matrix()
```

See [this notebook](https://github.com/dim02/svdUnfold/blob/master/example/example.ipynb) for an example.
