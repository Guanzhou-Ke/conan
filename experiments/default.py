from typing import List
from typing_extensions import Literal

from pydantic import BaseModel


class Config(BaseModel):
    @property
    def class_name(self):
        return self.__class__.__name__


class Experiment(Config):
    ################################
    #    Generation Config.        #
    ################################
    # Network architecture. default `mlp`
    arch: Literal['cnn', 'mlp', 'alexnet'] = 'mlp'
    # Encoder output.
    hidden_dim: int = 288
    # Training epochs
    epochs: int = 100
    # The number of running.
    n_runs:int = 20
    # Batch size
    batch_size: int = 128
    # Training log directory.
    log_dir: str = './logs'
    # Optimizer setting, SGD or ADAM, default as adam.
    opt: Literal['sgd', 'adam'] = 'adam'
    # Training device.
    device: Literal['cuda', 'cpu'] = 'cuda'
    # Display training information, default as False.
    verbose: bool = False
    # Validation interval.
    validation_intervals: int = 1
    # Randomization seed.
    seed: int = 0
    # Input channels.
    input_channels: List = [1, 1]
    # learning rate
    lr: float = 1e-3
    # Extract the training history
    extra_record: bool = False
    # Extract the hidden by an interval epoch
    extra_hidden: bool = False
    #
    extra_hidden_intervals: int = 5
    ################################
    #    Dataset Config.           #
    ################################
    # Dataset name
    ds_name: Literal['emnist', 'fmnist', 'coil-20', 'coil-100', 'voc', 'rgbd'] = None
    # Image size.
    img_size: int = 28
    ################################
    #    Clustering Module         #
    ################################
    # Clustering loss type
    clustering_loss_type: Literal['ddc', 'dec'] = 'ddc'
    # clustering hidden dim.
    cluster_hidden_dim: int = 128
    # The number of cluster
    num_cluster: int = 10
    ################################
    #    Fusion Module             #
    ################################
    # The number of view, Linear(in_feature * views, in_features)
    views: int = 2
    # The number of layer of fusion module.
    fusion_layers = 2
    # Fusion layer non-linear activation function.
    fusion_act: Literal['relu', 'sigmoid', 'tanh'] = 'relu'
    # Whether add BatchNormalize into fusion module.
    use_bn: bool = True
    ################################
    #    Contrastive Module        #
    ################################
    # Enable contrastive module, just for ablation study, default as True.
    enable_contrastive = True
    # Contrastive Module, default as simsiam.
    contrastive_type: Literal['simclr', 'simsiam'] = 'simclr'
    # Projection layers.
    projection_layers:int = 2
    # Projection dimension
    projection_dim: int = 128
    # prediction hidden dimension.
    prediction_hidden_dim: int = 128
    # Contrastive loss weight
    contrastive_lambda: float = 1.0
    # temperature coef.
    temperature = 0.1