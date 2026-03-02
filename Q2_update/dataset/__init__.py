"""Dataset generation package for reproducible synthetic benchmark data."""

from .generate_dataset import (
	generate_dataset,
	observation_mean,
	observation_mean_tf,
	transition_drift,
	transition_drift_tf,
)

__all__ = [
	"generate_dataset",
	"transition_drift",
	"observation_mean",
	"transition_drift_tf",
	"observation_mean_tf",
]
