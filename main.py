import torch
import random
import utils
import torch.backends.cudnn as cudnn
import h5py as h5
from dataloader import Dataloaders, DannDataset, DatasetBase
from torch.utils.data import DataLoader, Subset
from torch.optim import SGD, Adam
from models import RFFI, DomainDiscriMADA, linear2_dr2_bn
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from tllib.modules.grl import GradientReverseFunction, WarmStartGradientReverseLayer
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd


def __balance_val_split(dataset, val_split=0.):
    targets = np.array(dataset.label)
    train_indices, val_indices = train_test_split(
        np.arange(targets.shape[0]),
        test_size=val_split,
        stratify=targets
    )
    train_dataset = Subset(dataset, indices=train_indices)
    val_dataset = Subset(dataset, indices=val_indices)
    return train_dataset, val_dataset

def lr_schedule_step(optimizer, p):
    for param_group in optimizer.param_groups:
        # Update Learning rate
        param_group["lr"] = param_group["lr_mult"] * 0.01 / (
                1 + 10 * p) ** 0.75
        # Update weight_decay
        param_group["weight_decay"] = 2.5e-5 * param_group["decay_mult"]



def dann():
    epoches = 60
    batch_size = 128
    num_classes = 8
    num_domains = 4
    transform = 'cfo_fde_40cbw'
    finetune = False
    l2_norm = True
    lr = 0.01
    grl_hi = 1.


    f = h5.File('dg_dataset.h5')
    h5_group = f['5']

    loss_class_entropy = torch.nn.CrossEntropyLoss()
    loss_domain = torch.nn.CrossEntropyLoss()

    dataset = DannDataset(h5_group, transform=transform)
    train_set, val_set = __balance_val_split(dataset, 0.98)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    classifier = RFFI(num_classes=num_classes, finetune=finetune, l2_norm=l2_norm).to(device)

    if finetune:
         for param in classifier.backbone.parameters():
            param.requires_grad = False

    domain_discri = linear2_dr2_bn(512, output_size=num_domains).to(device)

    if 0:
        classifier.load_state_dict(torch.load('models/classifier0+1+3+4_l2norm_SGD_1000sample.pth'))

    discri_params = [{'params': domain_discri.parameters(), "lr_mult": 1.0, "lr": lr, 'way_mult': 2}]

    optimizer = SGD(classifier.get_parameters(base_lr=lr) + discri_params, momentum=0.9, weight_decay=1e-3, nesterov=True)
    #optimizer = Adam(classifier.get_parameters(base_lr=lr) + discri_params)

    lr_scheduler = ExponentialLR(optimizer, gamma=0.9)
    grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=grl_hi, max_iters=1000, auto_step=True)

    tot_losses = []
    class_losses = []
    domain_losses = []
    train_acc = []

    domain_acc = []
    val_acc = []

    best_acc = 0.0

    len_train_loader = len(train_loader)
    len_train_dataset = len(train_loader.dataset)
    len_val_loader = len(val_loader)
    len_val_dataset = len(val_loader.dataset)
    for epoch in range(epoches):
        classifier.train()
        domain_discri.train()

        running_tot_losses = 0.0
        running_class_losses = 0.0
        running_domain_losses = 0.0

        running_train_acc = 0.0
        running_domain_acc = 0.0

        lr = [optimizer.param_groups[i]['lr'] for i in range(len(optimizer.param_groups))]
        print(lr)

        with tqdm(total=len_train_loader) as pbar:
            for i in range(len_train_loader):
                x, y, d = next(iter(train_loader))

                logit, f = classifier(x.to(device))

                reverse_features = grl(f)
                logit_domain = domain_discri(reverse_features)

                loss_class = loss_class_entropy(logit, y.to(device))
                losses_domain = loss_domain(logit_domain, d.to(device))
                losses_domain = 0
                if finetune:
                    loss_tot = loss_class
                else:
                    loss_tot = loss_class + losses_domain

                y_pred = torch.max(logit, dim=1)[1]
                running_train_acc += torch.eq(y_pred, y.to(device)).sum().item()

                domain_pred = torch.max(logit_domain, dim=1)[1]
                running_domain_acc += torch.eq(domain_pred, d.to(device)).sum().item()

                optimizer.zero_grad()
                loss_tot.backward()
                optimizer.step()

                running_tot_losses += loss_tot.item()
                running_class_losses += loss_class.item()
                running_domain_losses += losses_domain.item()

                pbar.update(1)

        running_acc = 0.0
        classifier.eval()
        domain_discri.eval()


        with tqdm(total=len_val_loader) as pbar:
            for i in range(len_val_loader):
                x, y, d = next(iter(val_loader))
                logit = classifier(x.to(device))
                y_pred = torch.max(logit, dim=1)[1]

                running_acc += torch.eq(y_pred, y.to(device)).sum().item()

                pbar.update(1)


        lr_scheduler.step()

        tot_losses.append(running_tot_losses / len_train_dataset)
        class_losses.append(running_class_losses / len_train_dataset)
        domain_losses.append(running_domain_losses / len_train_dataset)
        val_acc.append(running_acc / len_val_dataset)
        domain_acc.append(running_domain_acc / len_train_dataset)
        train_acc.append(running_train_acc / len_train_dataset)

        print(
            '\n[Epoch %d] tot_loss: %.3f class_loss: %.3f domain_loss: %.3f train_acc: %.3f domain_acc: %.3f val_acc: '
            '%.3f' %
            (epoch, tot_losses[-1], class_losses[-1], domain_losses[-1], train_acc[-1], domain_acc[-1], val_acc[-1]))

        if val_acc[-1] >= best_acc:
            best_acc = val_acc[-1]
            torch.save(classifier.backbone.state_dict(), os.path.join('models', 'backbone0+1+3.pth'))
            torch.save(classifier.state_dict(), os.path.join('tclassifier0+1+3+4_l2norm_SGD_1000sample_finetune5.pth'))

    plt.figure(figsize=(10, 6))


    plt.plot(train_acc, label='Training Accuracy', marker='o')
    plt.plot(domain_acc, label='Domain Accuracy', marker='s')
    plt.plot(val_acc, label='Validation Accuracy', marker='^')


    plt.title('DANN Model Training Process')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()


    plt.grid(True)
    plt.show()



def test():
    batch_size = 64
    transform = 'cfo_fde_40cbw'

    classifier = RFFI(num_classes=8, l2_norm=True).to(device)
    classifier.eval()

    classifier.load_state_dict(torch.load('tclassifier0+1+3+4_l2norm_SGD_1000sample_finetune5.pth'))

    f = h5.File('dg_dataset.h5')
    h5_group = f['5']

    dataset = DatasetBase(h5_group, transform)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    y_test, y_pred = [], []

    with torch.no_grad():
        for data, label in test_loader:
            output = classifier(data.to(device)).cpu()
            predict = torch.max(output, 1)[1]

            y_pred.extend(predict.numpy())
            y_test.extend(label.cpu().numpy())

    ac = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    cr = classification_report(y_test, y_pred)

    print(ac)

    print(cr)

    labels = pd.DataFrame(cm).applymap(lambda v: f"{v}" if v != 0 else f"")
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=labels, cbar=True, cmap='Blues', fmt='s', linewidths=0.1)

    plt.xticks(np.arange(len(cm)) + 0.5, labels=[f"STA{i}" for i in range(len(cm))])
    plt.yticks(np.arange(len(cm)) + 0.5, labels=[f"STA{i}" for i in range(len(cm))], rotation=0)

    #plt.savefig('twoday-20230721-030735-fold-0.png')
    #plt.savefig('cm33.pdf', format='pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':

    device = torch.device("cuda")

    if 0:
        random.seed(42)
        torch.manual_seed(42)
        cudnn.deterministic = True

    test()
