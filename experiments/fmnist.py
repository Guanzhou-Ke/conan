from .default import Experiment

fmnist = Experiment(
    arch='cnn',
    hidden_dim=288,
    verbose=True,
    log_dir='./logs/mytest',
    device='cuda',
    extra_record=True,
    opt='adam',
    epochs=100,
    lr=1e-3,
    batch_size=100,
    cluster_hidden_dim=100,
    ds_name='fmnist',
    img_size=28,
    views=2,
    clustering_loss_type='ddc',
    num_cluster=10,
    fusion_act='relu',
    use_bn=True,
    contrastive_type='simclr',
    projection_layers=2,
    projection_dim=288,
    prediction_hidden_dim=0,  # Just for simsiam.
    contrastive_lambda=0.01,
    temperature=0.1,
    seed=0,
)
