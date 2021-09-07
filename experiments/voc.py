from .default import Experiment

voc = Experiment(
    arch='mlp',
    hidden_dim=256,
    verbose=False,
    log_dir='./logs/voc/simclr',
    device='cuda',
    extra_record=True,
    opt='adam',
    epochs=100,
    lr=1e-3,
    batch_size=100,
    cluster_hidden_dim=100,
    ds_name='voc',
    input_channels=[512, 399],
    views=2,
    clustering_loss_type='ddc',
    num_cluster=20,
    fusion_act='relu',
    use_bn=True,
    contrastive_type='simclr',
    projection_layers=2,
    projection_dim=256,
    prediction_hidden_dim=0,  # Just for simsiam.
    contrastive_lambda=0.01,
    temperature=0.1,
    seed=0,
)
