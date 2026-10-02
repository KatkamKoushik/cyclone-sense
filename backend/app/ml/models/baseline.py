from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class BaselineClimatologyPersistenceModel:
    """
    Physical Climatology and Persistence (CLIPER) baseline model for tropical cyclones.
    Uses closed-form regularized Ridge regression for intensity estimation and
    multinomial decision boundary for pattern severity classification.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.weights_intensity: Optional[np.ndarray] = None
        self.bias_intensity: float = 0.0
        self.weights_cat: Optional[np.ndarray] = None
        self.bias_cat: Optional[np.ndarray] = None
        self.mean_x: Optional[np.ndarray] = None
        self.std_x: Optional[np.ndarray] = None
        self.num_classes = 5
        self.is_fitted = False

    def fit(self, X: np.ndarray, y_intensity: np.ndarray, y_category: np.ndarray) -> "BaselineClimatologyPersistenceModel":
        """
        Fits baseline linear models on tabular environmental features.
        """
        n_samples, n_features = X.shape

        # Standardize features
        self.mean_x = np.mean(X, axis=0)
        self.std_x = np.std(X, axis=0)
        self.std_x[self.std_x < 1e-6] = 1.0
        X_norm = (X - self.mean_x) / self.std_x

        # Ridge regression closed-form: w = (X^T X + alpha * I)^(-1) X^T y
        X_bias = np.hstack([np.ones((n_samples, 1)), X_norm])
        reg_matrix = self.alpha * np.eye(n_features + 1)
        reg_matrix[0, 0] = 0.0  # Do not regularize intercept

        w_int = np.linalg.solve(X_bias.T @ X_bias + reg_matrix, X_bias.T @ y_intensity)
        self.bias_intensity = float(w_int[0])
        self.weights_intensity = w_int[1:]

        # One-vs-rest / multi-output ridge for categories
        y_cat_onehot = np.zeros((n_samples, self.num_classes), dtype=np.float32)
        for i in range(self.num_classes):
            y_cat_onehot[:, i] = (y_category == i).astype(np.float32)

        w_cat = np.linalg.solve(X_bias.T @ X_bias + reg_matrix, X_bias.T @ y_cat_onehot)
        self.bias_cat = w_cat[0, :]
        self.weights_cat = w_cat[1:, :]

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (predicted_intensity_knots, predicted_category_indices).
        """
        if not self.is_fitted or self.weights_intensity is None or self.weights_cat is None:
            raise RuntimeError("Model must be fitted before predict() is called.")

        X_norm = (X - self.mean_x) / self.std_x
        pred_intensity = X_norm @ self.weights_intensity + self.bias_intensity
        # Physical lower bound: tropical cyclones have positive wind speed
        pred_intensity = np.clip(pred_intensity, 10.0, 200.0)

        cat_logits = X_norm @ self.weights_cat + self.bias_cat
        pred_category = np.argmax(cat_logits, axis=1)

        return pred_intensity, pred_category

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns softmax-normalized multiclass probability estimates derived from Ridge classification logits.
        """
        if not self.is_fitted or self.weights_cat is None:
            raise RuntimeError("Model must be fitted before predict_proba() is called.")
        X_norm = (X - self.mean_x) / self.std_x
        cat_logits = X_norm @ self.weights_cat + self.bias_cat
        exp_logits = np.exp(cat_logits - np.max(cat_logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    def get_feature_importances(self, feature_names: List[str]) -> Dict[str, float]:
        """Returns normalized absolute weight magnitudes for environmental covariates."""
        if not self.is_fitted or self.weights_intensity is None:
            raise RuntimeError("Model is not fitted.")
        abs_weights = np.abs(self.weights_intensity)
        total = np.sum(abs_weights)
        if total > 0:
            norm_w = abs_weights / total
        else:
            norm_w = abs_weights
        return {name: round(float(w), 4) for name, w in zip(feature_names, norm_w)}
