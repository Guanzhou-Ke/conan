import json
import logging
import os

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from tabulate import tabulate

from util import measure_cluster, seed_everything, print_network
from experiments import get_experiment_config
from models import CONAN
from datatool import load_dataset


def get_current_labels(train_loader, model, device):
    model.eval()
    labels = []
    for data in train_loader:
        # measure data loading time
        Xs = [d.to(device) for d in data[0]]
        labels.append(model.predict(Xs).detach().cpu())
    labels = torch.cat(labels).long()
    return labels


def train_step(train_loader, model, epoch, device, verbose=1):
    model.train()
    tot_losses = []
    con_losses = []
    clu_losses = []
    if verbose:
        pbar = tqdm(total=len(train_loader), ncols=0, unit=" batch")
    for data in train_loader:
        # measure data loading time
        Xs = [d.to(device) for d in data[0]]
        model.optimizer.zero_grad()
        tot_loss, clu_loss, con_loss = model.get_loss(Xs)
        tot_losses.append(tot_loss.item())
        con_losses.append(con_loss.item())
        clu_losses.append(clu_loss.item())
        # compute gradient and do SGD step
        tot_loss.backward()
        model.optimizer.step()
        if verbose:
            pbar.update()
            pbar.set_postfix(
                epoch=epoch,
                total_loss=f"{np.mean(tot_losses):.4f}",
                clustering_loss=f"{np.mean(clu_losses):.4f}",
                contrastive_loss=f"{np.mean(con_losses):.4f}",
            )
    if verbose:
        pbar.close()
    return np.mean(tot_losses), np.mean(clu_losses), np.mean(con_losses)


def save_dict(obj, path):
    try:
        with open(path, 'w') as f:
            save_dict = {}
            for key in obj.keys():
                if isinstance(obj[key], list):
                    save_dict[key] = obj[key]
                elif isinstance(obj[key], int):
                    save_dict[key] = obj[key]
                elif isinstance(obj[key], np.ndarray):
                    save_dict[key] = obj[key].tolist()
            json.dump(save_dict, f, indent=4)
            print(f'Saved dict at {path}')
    except Exception as e:
        print(e)


class Recoder:

    def __init__(self):
        self.epoch = []
        self.total_losses = []
        self.contrastive_losses = []
        self.clustering_losses = []
        self.accuracy = []
        self.nmi = []
        self.purity = []

    def batch_update(self, epoch, tot_loss, clu_loss, con_loss, acc, nmi, pur):
        self.epoch.append(epoch)
        self.total_losses.append(tot_loss)
        self.contrastive_losses.append(con_loss)
        self.clustering_losses.append(clu_loss)
        self.accuracy.append(acc)
        self.nmi.append(nmi)
        self.purity.append(pur)

    def to_dict(self):
        return {"epoch": self.epoch,
                "total_losses": self.total_losses,
                "contrastive_losses": self.contrastive_losses,
                "clustering_losses": self.clustering_losses,
                "accuracy": self.accuracy,
                "nmi": self.nmi,
                "purity": self.purity}


def main(model, dataset, args, run):
    ### Data loading ###
    num_workers = 8
    model.to(args.device)
    history = Recoder()
    train_loader = DataLoader(dataset, args.batch_size, num_workers=num_workers,
                              shuffle=True, drop_last=True)
    valid_loader = DataLoader(dataset, args.batch_size*2, num_workers=num_workers, shuffle=False)
    valid_loader.transform = None
    log_dir = os.path.join(args.log_dir, f'run_{run}')
    writer = SummaryWriter(log_dir=log_dir)
    logging.basicConfig(filename=os.path.join(log_dir, 'training.log'), level=logging.INFO)
    hparams_head = ['Hyper-parameters', 'Value']
    logging.info(tabulate(args.dict().items(), headers=hparams_head))
    targets = valid_loader.dataset.targets
    previous_label = None
    if isinstance(targets, list):
        targets = np.array(targets)
    elif isinstance(targets, torch.Tensor):
        targets = targets.numpy()
    else:
        raise ValueError('targets must be list, numpy or tensor.')
    best_loss = np.inf
    if args.clustering_loss_type == 'dec':
        print('[Initialize Centroids]...')
        kmeans = KMeans(n_clusters=args.num_cluster, n_init=20)
        model.train()
        features = []
        # form initial cluster centres
        for data in train_loader:
            Xs = [d.to(args.device) for d in data[0]]
            features.append(model.commonZ(Xs).detach().cpu())
        kmeans.fit(torch.cat(features).numpy())
        cluster_centers = torch.tensor(
            kmeans.cluster_centers_, dtype=torch.float, requires_grad=True
        )
        cluster_centers = cluster_centers.to(args.device)
        with torch.no_grad():
            # initialise the cluster centers
            model.clustering_module.cluster_centers.copy_(cluster_centers)
    print('[TRAIN]...')
    ###############################################################################
    for epoch in range(args.epochs):

        if args.extra_hidden and (epoch % args.extra_hidden_intervals == 0):
            model.eval()
            hs_l = {}
            for v in range(args.views):
                hs_l.setdefault(f"h{v+1}", list())
            z_l = []
            for data in train_loader:
                Xs = [d.to(args.device) for d in data[0]]
                hs, z = model.extract_all_hidden(Xs)
                for v in range(args.views):
                    hs_l[f'h{v+1}'].append(hs[v].detach().cpu())
                z_l.append(z.detach().cpu())
            z_l = torch.cat(z_l)
            torch.save({"hs": hs_l, "z":z_l}, os.path.join(log_dir, f'epoch_{epoch}_hidden.data'))

        # Supervised Training
        tot_loss_avg, clu_loss_avg, con_loss_avg = \
            train_step(train_loader, model, epoch+1, args.device, verbose=args.verbose)
        writer.add_scalar('training/total_loss', tot_loss_avg, global_step=epoch)
        writer.add_scalar('training/clustering_loss', clu_loss_avg, global_step=epoch)
        writer.add_scalar('training/contrastive_loss', con_loss_avg, global_step=epoch)

        if epoch % args.validation_intervals == 0:
            model.eval()
            predicted = get_current_labels(valid_loader, model, args.device).numpy()
            acc, nmi, pur = measure_cluster(predicted, targets)
            if previous_label is not None:
                nmi_t_1 = normalized_mutual_info_score(predicted, previous_label)
            else:
                nmi_t_1 = 0
            previous_label = predicted
            if args.verbose:
                values = [(epoch+1, acc, nmi, pur, nmi_t_1)]
                headers = ['Validation Epoch', 'Accuracy', 'NMI', 'Purity', 'nmi_(t-1)']
                print(tabulate(values, headers=headers))
            writer.add_scalar('validation/acc', acc, global_step=epoch)
            writer.add_scalar('validation/nmi', nmi, global_step=epoch)
            writer.add_scalar('validation/purity', pur, global_step=epoch)
            writer.add_scalar('validation/nmi_t_1', nmi_t_1, global_step=epoch)
            history.batch_update(epoch, tot_loss_avg, clu_loss_avg, con_loss_avg, acc, nmi, pur)

        if tot_loss_avg < best_loss:
            torch.save(model.state_dict(), os.path.join(log_dir, f'model_weight_best.pth'))
            best_loss = tot_loss_avg
            print(f"Saved model at {os.path.join(log_dir, f'model_weight_best.pth')}, best loss: {best_loss:.6f}.")

    writer.close()
    return history.to_dict()


if __name__ == '__main__':
    name, args = get_experiment_config()
    seed_everything(args.seed)
    dataset = load_dataset(args.ds_name, args.img_size)
    hparams_head = ['Hyper-parameters', 'Value']
    run_histories = []
    print(tabulate(args.dict().items(), headers=hparams_head))
    for run in range(args.n_runs):
        model = CONAN(args)
        print_network(model)
        history = main(model, dataset, args, run)
        run_histories.append(history)
    if args.extra_record:
        torch.save(run_histories, os.path.join(args.log_dir, 'records.his'))
        logging.info(f"Saved records at {os.path.join(args.log_dir, 'records.his')}")
