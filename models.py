import math
from typing import Optional

import torch
from torch import nn
from torch.nn import init
from torch.nn import Parameter

from backbones import NormalCNN, MLP, AlexNet
from loss_func import DECLoss, DDCLoss, SimSiamLoss, SimCLRLoss
from optimizer import Optimizer


class CONAN(nn.Module):
    def __init__(self, args):
        super(CONAN, self).__init__()
        self.args = args
        backbones = build_backbones(args)
        for idx, bb in enumerate(backbones):
            bb.to(args.device)
            self.__setattr__(f'encoder{idx}', bb)

        self.fusion_layer = FusionLayer(args)

        self.clustering_module, self.cls_criterion = build_clustering_module(args)
        if args.enable_contrastive:
            self.contrastive_module = build_contrastive_module(args)

        self.apply(self.weights_init('xavier'))
        self.optimizer = Optimizer(self.parameters(), lr=args.lr, opt=args.opt)

    def _get_hs(self, Xs):
        hs = [bb(x) for bb, x in zip([self.__getattr__(f"encoder{idx}") for idx in range(self.args.views)], Xs)]
        return hs

    def forward(self, Xs):
        hs = self._get_hs(Xs)
        z = self.fusion_layer(hs)
        if self.args.clustering_loss_type == 'dec':
            y = self.clustering_module(z)
        elif self.args.clustering_loss_type == 'ddc':
            y, _ = self.clustering_module(z)
        else:
            raise ValueError('Clustering loss error.')
        return y

    @torch.no_grad()
    def commonZ(self, Xs):
        hs = self._get_hs(Xs)
        z = self.fusion_layer(hs)
        return z

    @torch.no_grad()
    def extract_all_hidden(self, Xs):
        hs = self._get_hs(Xs)
        z = self.fusion_layer(hs)
        return hs, z

    def get_loss(self, Xs):
        hs = self._get_hs(Xs)
        z = self.fusion_layer(hs)
        if self.args.clustering_loss_type == 'dec':
            y = self.clustering_module(z)
            clustering_loss = self.cls_criterion(y)
        elif self.args.clustering_loss_type == 'ddc':
            y, h = self.clustering_module(z)
            clustering_loss = self.cls_criterion(y, h)
        else:
            raise ValueError('Clustering loss error.')
        if self.args.enable_contrastive:
            contrastive_loss = self.contrastive_module.get_loss(z, hs)
            tot_loss = clustering_loss + contrastive_loss
            return tot_loss, clustering_loss, contrastive_loss
        else:
            tot_loss = clustering_loss
            return tot_loss, clustering_loss, clustering_loss

    @torch.no_grad()
    def predict(self, Xs):
        return self(Xs).detach().cpu().max(1)[1]

    def weights_init(self, init_type='gaussian'):
        def init_fun(m):
            classname = m.__class__.__name__
            if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
                # print m.__class__.__name__
                if init_type == 'gaussian':
                    init.normal_(m.weight, 0.0, 0.02)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight, gain=math.sqrt(2))
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight, gain=math.sqrt(2))
                elif init_type == 'default':
                    pass
                else:
                    assert 0, "Unsupported initialization: {}".format(init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias, 0.0)

        return init_fun

def build_clustering_module(args):
    if args.clustering_loss_type == 'dec':
        clustering_layer = DECModule(cluster_number=args.num_cluster,
                                                  embedding_dimension=args.hidden_dim)
        cls_criterion = DECLoss()
    elif args.clustering_loss_type == 'ddc':
        clustering_layer = DDCModule(args.hidden_dim, args.cluster_hidden_dim, args.num_cluster)
        cls_criterion = DDCLoss(args.num_cluster, device=args.device)
    else:
        raise ValueError('Loss type must be dec or ddc.')
    return clustering_layer, cls_criterion


def build_contrastive_module(args):
    if args.contrastive_type == 'simsiam':
        return SimSiamModule(args)
    elif args.contrastive_type == 'simclr':
        return SimCLRModule(args)
    else:
        raise ValueError('Contrastive type error.')


def build_backbones(args):
    """
    build view-specific backbone.
    :param arch:
    :param input_channels:
    :param views:
    :param width:
    :param replace_fc:
    :return:
    """
    arch = args.arch
    input_channels = args.input_channels
    if arch == 'cnn':
        backbones = [NormalCNN(input_channels=_) for _ in input_channels]
    elif arch == 'mlp':
        backbones = [MLP(input_channels=_) for _ in input_channels]
    elif arch == 'alexnet':
        backbones = [AlexNet(input_channels=_) for _ in input_channels]
    else:
        raise ValueError('Architecture must be r18(resnet18), mlp or cnn(normal cnn).')

    for bb in backbones:
        bb.fc = nn.Identity()
    return backbones


class FusionLayer(nn.Module):

    def __init__(self, args):
        super(FusionLayer, self).__init__()
        act_func = args.fusion_act
        views = args.views
        use_bn = args.use_bn
        mlp_layers = args.fusion_layers
        in_features = args.hidden_dim
        if act_func == 'relu':
            self.act = nn.ReLU()
        elif act_func == 'tanh':
            self.act = nn.Tanh()
        elif act_func == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            raise ValueError('Activate function must be ReLU or Tanh.')
        self.layers = [self._make_layers(in_features * views, in_features, self.act, use_bn)]
        if mlp_layers > 1:
            layers = [self._make_layers(in_features, in_features,
                                        self.act if _ < (mlp_layers - 2) else nn.Identity(),
                                        use_bn if _ < (mlp_layers - 2) else False) for _ in range(mlp_layers - 1)]
            self.layers += layers
        self.layers = nn.Sequential(*self.layers)

    def forward(self, h):
        h = torch.cat(h, dim=-1)
        z = self.layers(h)
        return z

    def _make_layers(self, in_features, out_features, act, bn=False):
        layers = nn.ModuleList()
        layers.append(nn.Linear(in_features, out_features))
        layers.append(act)
        if bn:
            layers.append(nn.BatchNorm1d(out_features))
        return nn.Sequential(*layers)


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048, num_layers=2):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.num_layers = num_layers
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        if self.num_layers == 3:
            self.layer2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):  # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class SimSiamModule(nn.Module):

    def __init__(self, args):
        super(SimSiamModule, self).__init__()
        in_features = args.hidden_dim
        self.projection = projection_MLP(in_features,
                                         hidden_dim=args.projection_dim,
                                         out_dim=args.cluster_hidden_dim,
                                         num_layers=args.projection_layers)
        self.predictor = prediction_MLP(args.cluster_hidden_dim,
                                        hidden_dim=args.prediction_hidden_dim,
                                        out_dim=args.cluster_hidden_dim)
        self.con_criterion = SimSiamLoss()
        self.con_lambda = args.contrastive_lambda


    def forward(self, h):
        p = self.projection(h)
        z = self.predictor(p)
        return p, z

    def get_loss(self, ch, hs):
        cp, cz = self(ch)
        sub_loss = 0
        for h in hs:
            p, z = self(h)
            sub_loss += self.con_criterion(cp, p, cz, z)
        return self.con_lambda * sub_loss


class SimCLRModule(nn.Module):

    def __init__(self, args):
        super(SimCLRModule, self).__init__()
        in_features = args.hidden_dim
        if args.projection_layers == 0:
            self.projection = nn.Identity()
        else:
            self.projection = projection_MLP(in_features,
                                             hidden_dim=args.projection_dim,
                                             out_dim=args.cluster_hidden_dim,
                                             num_layers=args.projection_layers)
        self.con_criterion = SimCLRLoss(args)
        self.con_lambda = args.contrastive_lambda
        self.args = args

    def forward(self, h):
        h = self.projection(h)
        return h

    def get_loss(self, ch, hs):
        cp = self(ch)
        sub_loss = 0
        for h in hs:
            p = self(h)
            ps = torch.cat([cp, p], dim=-1)
            sub_loss += self.con_criterion(ps)
        return self.con_lambda * sub_loss


class DECModule(nn.Module):
    def __init__(
            self,
            cluster_number: int,
            embedding_dimension: int,
            alpha: float = 1.0,
            cluster_centers: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Module to handle the soft assignment, for a description see in 3.1.1. in Xie/Girshick/Farhadi,
        where the Student's t-distribution is used measure similarity between feature vector and each
        cluster centroid.

        :param cluster_number: number of clusters
        :param embedding_dimension: embedding dimension of feature vectors
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        :param cluster_centers: clusters centers to initialise, if None then use Xavier uniform
        """
        super(DECModule, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.cluster_number, self.embedding_dimension, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the soft assignment for a batch of feature vectors, returning a batch of assignments
        for each cluster.

        :param batch: FloatTensor of [batch size, embedding dimension]
        :return: FloatTensor [batch size, number of clusters]
        """
        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)


class DDCModule(nn.Module):

    def __init__(self, in_features, hidden_dim, num_cluster):
        super(DDCModule, self).__init__()

        self.hidden_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim, momentum=0.1)
        )

        self.clustering_layer = nn.Sequential(
            nn.Linear(hidden_dim, num_cluster),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        h = self.hidden_layer(x)
        y = self.clustering_layer(h)
        return y, h

