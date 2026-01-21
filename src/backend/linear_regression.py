"""
The base outline was obtained from Professor Artzi's Padlet - Starter code - regression app.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

class LinearRegression:
    """
    Linear Regression using Gradient Descent.

    This class implements linear regression from scratch using
    gradient descent optimization.

    Model: y = Xw + b
    Loss: MSE = (1/n) * sum((y - y_pred)^2)
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Initialize the model.

        Args:
            learning_rate: Step size for gradient descent (α)
            n_iterations: Number of training iterations
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []  # Track loss over iterations

    def fit(self, X, y):
        """
        Train the model using gradient descent.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
        """
        n_samples, n_features = X.shape

        self.weights = np.random.randn(n_features)
        self.bias = 0

        # Training loop
        for iteration in range(self.n_iterations):

            y_pred = X @ self.weights + self.bias


            loss = np.mean(((y - y_pred)**2))

            self.losses.append(loss)

            dw = -(2/n_samples) * X.T @ (y - y_pred)
            db = -(2/n_samples) * sum(y - y_pred)

            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Loss = {loss:.4f}")

    def predict(self, X):
        """
        Make predictions on new data.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            y_pred: Predictions (n_samples,)
        """

        y_pred = X @ self.weights + self.bias
        return y_pred


    # ==========================================
    # EVALUATION METRICS
    # ==========================================

    def compute_metrics(self, y_true, y_pred):
        """
        Compute evaluation metrics for regression.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary with metrics
        """

        n = len(y_true)

        mse = (1/n) * sum((y_true - y_pred)** 2)

        rmse = np.sqrt(mse)

        mae = (1/n) * sum(np.abs(y_true-y_pred))

        ss_res = sum((y_true-y_pred)**2)
        y_mean = np.average(y_true)
        ss_tot = sum((y_true - y_mean)**2)
        r2 = 1 - (ss_res / ss_tot)

        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }


    # ==========================================
    # VISUALIZATION FUNCTIONS
    # ==========================================

    def plot_training_progress(self):
        """
        Plot loss vs. iteration to show training progress.

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=list(range(len(self.losses))),
            y=self.losses,
            mode='lines',
            name="losses"
        ))

        fig.update_layout(
            title='Training Progress: Loss vs Iteration',
            xaxis_title='Iteration',
            yaxis_title='Loss (MSE)',
            template='plotly_white',
            height=400
        )

        return fig


    def plot_predictions(self,X, y_true, y_pred, y_actual=None):
        # 1. Flatten X to a 1D array
        X_flat = X.flatten()
        
        # 2. Get the indices that would sort X_flat
        sort_idx = np.argsort(X_flat)
        
        # 3. Reorder X and all Y arrays using those indices
        X_sorted = X_flat[sort_idx]
        y_true_sorted = y_true[sort_idx]
        y_pred_sorted = y_pred[sort_idx]
        
        fig = go.Figure()

        if y_actual is not None:
            y_actual_sorted = y_actual[sort_idx]
            fig.add_trace(go.Scatter(
                x=X_sorted,
                y=y_actual_sorted,
                mode="markers",
                name="Actual Data (Noisy)",
                marker=dict(color='rgba(152, 0, 0, .8)', size=6)
            ))

        fig.add_trace(go.Scatter(
            x=X_sorted,
            y=y_true_sorted,
            mode="lines",
            name="True Function",
            line=dict(color='RoyalBlue', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=X_sorted,
            y=y_pred_sorted,
            mode="lines",
            name="Model Prediction",
            line=dict(color='FireBrick', width=2, dash='dash')
        ))

        fig.update_layout(
            title='Model Predictions vs Actual Data',
            xaxis_title='X',
            yaxis_title='y',
            template='plotly_white',
            height=500
        )

        return fig


    def plot_residuals(self, y_true, y_pred):
        """
        Plot residuals (errors) to check model fit.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Plotly figure
        """

        residuals = y_true - y_pred

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=y_pred,
            y=residuals,
            mode="markers",
            name="Residuals"
        ))
        fig.add_hline(y=0)

        fig.update_layout(
            title='Residual Plot: Check for Patterns',
            xaxis_title='Predicted Value',
            yaxis_title='Residual (True - Predicted)',
            template='plotly_white',
            height=400
        )

        return fig

    def plot_regression_line(self, train_df, test_df):
        """
        Plot the regression line along with training and testing data points.

        Args:
            train_df: Training DataFrame
            test_df: Testing DataFrame
        """

        fig = go.Figure()

        # Training data points
        fig.add_trace(go.Scatter(
            x=train_df["X"],
            y=train_df["y"],
            mode="markers",
            name="Training Data",
            marker=dict(color='blue', size=6)
        ))

        # Testing data points
        fig.add_trace(go.Scatter(
            x=test_df["X"],
            y=test_df["y"],
            mode="markers",
            name="Testing Data",
            marker=dict(color='orange', size=6)
        ))

        # Regression line
        all_X = pd.concat([train_df["X"], test_df["X"]])
        X_range = np.linspace(all_X.min(), all_X.max(), 100).reshape(-1, 1)
        y_range_pred = self.predict(X_range)

        fig.add_trace(go.Scatter(
            x=X_range.flatten(),
            y=y_range_pred,
            mode="lines",
            name="Regression Line",
            line=dict(color='red', width=2)
        ))

        fig.update_layout(
            title='Linear Regression Fit',
            xaxis_title='X',
            yaxis_title='y',
            template='plotly_white',
            height=500
        )

        st.plotly_chart(fig)

