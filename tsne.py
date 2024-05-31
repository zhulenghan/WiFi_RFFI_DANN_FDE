import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from models import RFFI
import torch.nn as nn
import h5py as h5
from dataloader import DatasetBase
from main import __balance_val_split



model_save_path = 'models/classifier0+1+2+3_l2norm_SGD_1000sample.pth'
transform = 'cfo_fde_40cbw'
rffi = RFFI(8, l2_norm=True).cuda()
rffi.load_state_dict(torch.load(model_save_path))
#model = nn.Sequential(*list(rffi.children())[:-1], *list(list(rffi.children())[-1].children())[:-1])

model = nn.Sequential(*list(rffi.children())[:-1])
model = rffi.backb

model_save_path = 'models/classifier0+1+2+3_l2norm_SGD_1000sample.pth'
transform = 'cfo_fde_40cbw'
rffi = RFFI(8, l2_norm=True).cuda()
rffi.load_state_dict(torch.load(model_save_path))


model.eval()

f = h5.File('dg_dataset.h5')
h5_group1 = f['4']

h5_group2 = f['0']


dataset1 = DatasetBase(h5_group1, transform)
dataset1, _ = __balance_val_split(dataset1, 0.4)

dataset2 = DatasetBase(h5_group2, transform)
dataset2, _ = __balance_val_split(dataset2, 0.4)



def extract_features(dataset):
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    features = []

    labels = []

    with torch.no_grad():
        for inputs, label in loader:
            inputs = inputs.to('cuda')

            output = model(inputs).cpu().numpy()
            inputs = torch.flatten(inputs, 1)
            features.append(inputs.cpu().numpy())

            labels.append(label.numpy())

    features = np.concatenate(features, axis=0)

    labels = np.concatenate(labels, axis=0)

    return features, labels



features1, labels1 = extract_features(dataset1)

features2, labels2 = extract_features(dataset2)



features = np.concatenate([features1, features2], axis=0)

labels = np.concatenate([labels1, labels2], axis=0)

dataset_labels = np.concatenate([np.zeros_like(labels1), np.ones_like(labels2)])
tsne = TSNE(n_components=2, random_state=42)
markers = ['o', 'x']  
features_2d = tsne.fit_transform(features)



unique_dataset_labels = np.unique(dataset_labels)

dataset_colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_dataset_labels)))

plt.figure(figsize=(7, 7)) 

for dataset_label, dataset_color, marker in zip(unique_dataset_labels, dataset_colors, markers):

    dataset_indices = dataset_labels == dataset_label

    unique_labels = np.unique(labels[dataset_indices])

    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        indices = np.logical_and(dataset_indices, labels == label)

        plt.scatter(features_2d[indices, 0], features_2d[indices, 1], c=[color], marker=marker,
                    label=f'Dataset {int(dataset_label) + 1}, Class {label + 1}', alpha=0.5)

plt.tick_params(axis='both', which='both', length=0, labelleft=False, labelbottom=False)

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='gray', marker='o', linestyle='None'),
                Line2D([0], [0], color='gray', marker='x', linestyle='None')]

plt.legend(custom_lines, ['Dataset 0 (source)', 'Dataset 4 (target)'])
import matplotlib.patches as patches
circle_center = (45, -25)  
circle_radius = 25  
circle = patches.Circle(circle_center, circle_radius, linewidth=2, edgecolor='grey', facecolor='none')
plt.gca().add_patch(circle)

circle_center = (70, -10)  
circle_radius = 20  
circle = patches.Circle(circle_center, circle_radius, linewidth=2, edgecolor='grey', facecolor='none')
plt.gca().add_patch(circle)

plt.text(70, -10, 'target', horizontalalignment='center')
plt.text(45, -25, 'source', horizontalalignment='center')

plt.tight_layout()
#plt.savefig('tsne_dann.pdf', format='pdf', bbox_inches='tight')
plt.show()
