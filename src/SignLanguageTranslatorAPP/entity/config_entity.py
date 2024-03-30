from dataclasses import dataclass
from pathlib import Path
import tensorflow as tf

@dataclass(frozen=True)
class LandmarksExtractionConfig:
    root_dir: Path
    output_dir: Path

@dataclass(frozen=True)
class PreprocessingConfig:
    root_dir: Path
    params_FIXED_FRAMES: tf.int32
    params_ROWS_PER_FRAME: tf.int32
    params_UNITS: tf.int32
    

@dataclass(frozen=True)
class ModelConfig:
    root_dir: Path
    model_weight_dir: Path
    params_UNITS: tf.int32
    params_NUM_BLOCKS: tf.int32
    params_MLP_RATIO: tf.int32
    params_EMBEDDING_DROPOUT: tf.float32
    params_MLP_DROPOUT_RATIO: tf.float32
    params_CLASSIFIER_DROPOUT_RATIO: tf.float32
    params_N_EPOCHS: tf.int32