import torch
import random
import utils
import torch.backends.cudnn as cudnn
import h5py as h5
from dataloader import Dataloaders, DannDataset, DatasetBase
from torch.utils.data import DataLoader, Subset
from torch.optim import SGD, Adam
from models import RFFI, DomainDiscriMADA, linear2_dr2_bn, AlexNet1D
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


def cnn():
    epoches = 60
    batch_size = 128
    num_classes = 8
    num_domains = 4
    transform = 'legacy_preamble'
    finetune = False
    l2_norm = True
    lr = 0.001
    grl_hi = 1.

    f = h5.File('dg_dataset.h5')
    h5_group = f['0+1+2+4']

    loss_class_entropy = torch.nn.CrossEntropyLoss()

    dataset = DatasetBase(h5_group, transform=transform)
    train_set, val_set = __balance_val_split(dataset, 0.1)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    classifier = AlexNet1D(num_classes=num_classes).to(device)

    domain_discri = linear2_dr2_bn(512, output_size=num_domains).to(device)

    if 0:
        classifier.load_state_dict(torch.load('models/classifier0+1+2+3_l2norm_SGD_1000sample.pth'))

    optimizer = Adam(classifier.parameters(), lr=lr)
    lr_scheduler = ExponentialLR(optimizer, gamma=0.9)

    tot_losses = []
    class_losses = []
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


        with tqdm(total=len_train_loader) as pbar:
            for i in range(len_train_loader):
                x, y = next(iter(train_loader))

                logit = classifier(x.to(device))

                loss_class = loss_class_entropy(logit, y.to(device))
                loss_tot = loss_class

                y_pred = torch.max(logit, dim=1)[1]
                running_train_acc += torch.eq(y_pred, y.to(device)).sum().item()


                optimizer.zero_grad()
                loss_tot.backward()
                optimizer.step()

                running_tot_losses += loss_tot.item()
                running_class_losses += loss_class.item()

                pbar.update(1)

        running_acc = 0.0
        classifier.eval()

        with tqdm(total=len_val_loader) as pbar:
            for i in range(len_val_loader):
                x, y = next(iter(val_loader))
                logit = classifier(x.to(device))
                y_pred = torch.max(logit, dim=1)[1]

                running_acc += torch.eq(y_pred, y.to(device)).sum().item()

                pbar.update(1)

        lr_scheduler.step()

        tot_losses.append(running_tot_losses / len_train_dataset)
        class_losses.append(running_class_losses / len_train_dataset)
        val_acc.append(running_acc / len_val_dataset)
        train_acc.append(running_train_acc / len_train_dataset)

        print(
            '\n[Epoch %d] tot_loss: %.3f class_loss: %.3f train_acc: %.3f val_acc: '
            '%.3f' %
            (epoch, tot_losses[-1], class_losses[-1], train_acc[-1], val_acc[-1]))

        if val_acc[-1] >= best_acc:
            best_acc = val_acc[-1]
            # torch.save(classifier.backbone.state_dict(), os.path.join('models', 'backbone0+1+3.pth'))
            torch.save(classifier.state_dict(), os.path.join('models', 'iqcnn0+1+2+4.pth'))

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

def draw():
    groups = ['0,1,2,3->4', '1,2,3,4->0', '0,2,3,4->1', '0,1,3,4->2', '0,1,2,4->3']
    models = ['I/Q CNN', 'FDE+DANN', 'FDE+DANN (finetuned)']
    performance = [
        [21.18, 36.41, 11.88, 0.4, 15.91],
        [60.04, 55.21, 44, 60.54, 59.54],
        [78.13, 92, 86.20, 84.86, 80.84]
    ]


    n_groups = 5
    fig, ax = plt.subplots(figsize=(10, 4))
    index = np.arange(n_groups)
    bar_width = 0.2
    opacity = 0.8


    rects = []
    for i, model_performance in enumerate(performance):
        rects.append(plt.bar(index + i * bar_width, model_performance, bar_width, alpha=opacity, label=models[i]))


    for rect in rects:
        for bar in rect:
            height = bar.get_height()
            ax.annotate('{}'.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 1),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    plt.xlabel('Datasets Generalization Task')
    plt.ylabel('Accuracy (%)')
    plt.xticks(index + bar_width, groups)
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.savefig('acc.pdf', format='pdf', bbox_inches='tight')
    plt.show()


def test():
    batch_size = 64
    transform = 'legacy_preamble'

    classifier = AlexNet1D(num_classes=8).to(device)
    classifier.eval()

    classifier.load_state_dict(torch.load('models/iqcnn1+2+3+4.pth'))

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


    labels = pd.DataFrame(cm).applymap(lambda v: f"{v}" if v != 0 else f"")
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=labels, cbar=False, cmap='Blues', fmt='s', linewidths=0.1)

    plt.xticks(np.arange(len(cm)) + 0.5, labels=[f"STA{i}" for i in range(len(cm))])
    plt.yticks(np.arange(len(cm)) + 0.5, labels=[f"STA{i}" for i in range(len(cm))], rotation=0)

    # plt.savefig('twoday-20230721-030735-fold-0.png')
    plt.savefig('cm31.pdf', format='pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if 0:
        random.seed(42)
        torch.manual_seed(42)
        cudnn.deterministic = True

    test()
