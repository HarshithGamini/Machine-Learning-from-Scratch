import numpy as np

class PCA:
    
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        
    def fit(self, X):
        # Mean Centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        
        # Covariance, Function needs samples as columns
        cov = np.cov(X.T)
        
        # Eigenvectors, Eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # Eigenvectors v = [:, i] column vector, transpose this for easier calculations
        eigenvectors = eigenvectors.T

        # sort eigenvectors
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        self.components = eigenvectors[:self.n_components]
        
    def transform(self, X):
        # Project data into Lower Dimension
        X = X - self.mean
        return np.dot(X, self.components.T)
