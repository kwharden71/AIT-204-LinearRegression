"""
Synthetic Dataset Generators for Regression Tasks
AIT-204 Topic 1: Background Math and Gradient-Based Learning

This module provides various synthetic dataset generation functions
suitable for regression model training and gradient descent learning.
Each function creates data with known mathematical relationships,
allowing students to understand how models learn from data.
"""

"""
This is from Professor Artzi's git repo:
https://github.com/isac-artzi/AIT-204-Topic1/blob/main/data_generators.py
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


class SyntheticDataGenerator:
    """
    A comprehensive class for generating various types of synthetic datasets
    suitable for regression tasks in deep learning courses.

    This generator is designed to help students understand:
    - Linear and non-linear relationships
    - Effect of noise on data
    - Feature scaling and normalization
    - Multi-dimensional input spaces
    - Polynomial features and complexity
    """

    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize the data generator with optional random seed for reproducibility.

        Args:
            random_seed: Integer seed for numpy random number generator.
                        If None, results will vary each time.
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        self.random_seed = random_seed

    def generate_simple_linear(self,
                               n_samples: int = 100,
                               slope: float = 2.0,
                               intercept: float = 1.0,
                               noise_std: float = 0.5,
                               x_range: Tuple[float, float] = (0, 10)) -> pd.DataFrame:
        """
        Generate simple linear regression data: y = mx + b + ε

        This is the fundamental regression problem where the relationship
        between input (x) and output (y) is linear. Perfect for understanding
        gradient descent optimization.

        Args:
            n_samples: Number of data points to generate
            slope: Slope (m) of the linear relationship
            intercept: Y-intercept (b) of the line
            noise_std: Standard deviation of Gaussian noise (ε)
            x_range: Tuple of (min, max) for x values

        Returns:
            DataFrame with columns ['x', 'y', 'y_true'] where y_true is noiseless output
        """
        # Generate uniformly distributed x values
        x = np.random.uniform(x_range[0], x_range[1], n_samples)

        # Calculate true y values (without noise)
        y_true = slope * x + intercept

        # Add Gaussian noise to create realistic data
        noise = np.random.normal(0, noise_std, n_samples)
        y = y_true + noise

        # Create DataFrame for easy manipulation and export
        df = pd.DataFrame({
            'x': x,
            'y': y,
            'y_true': y_true
        })

        return df

    def generate_multiple_linear(self,
                                 n_samples: int = 100,
                                 n_features: int = 3,
                                 coefficients: Optional[np.ndarray] = None,
                                 intercept: float = 5.0,
                                 noise_std: float = 1.0,
                                 feature_range: Tuple[float, float] = (-5, 5)) -> pd.DataFrame:
        """
        Generate multiple linear regression data: y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b + ε

        This represents multi-dimensional input space, crucial for understanding
        gradient descent in higher dimensions and partial derivatives.

        Args:
            n_samples: Number of data points to generate
            n_features: Number of input features
            coefficients: Array of weights for each feature. If None, random weights used
            intercept: Bias term (b)
            noise_std: Standard deviation of Gaussian noise
            feature_range: Tuple of (min, max) for feature values

        Returns:
            DataFrame with columns ['x1', 'x2', ..., 'xn', 'y', 'y_true']
        """
        # Generate random coefficients if not provided
        if coefficients is None:
            coefficients = np.random.uniform(-3, 3, n_features)
        else:
            assert len(coefficients) == n_features, "Coefficients must match n_features"

        # Generate feature matrix (n_samples × n_features)
        X = np.random.uniform(feature_range[0], feature_range[1], (n_samples, n_features))

        # Calculate true output: matrix multiplication + intercept
        y_true = X @ coefficients + intercept

        # Add Gaussian noise
        noise = np.random.normal(0, noise_std, n_samples)
        y = y_true + noise

        # Create DataFrame with named columns
        df = pd.DataFrame(X, columns=[f'x{i+1}' for i in range(n_features)])
        df['y'] = y
        df['y_true'] = y_true

        return df