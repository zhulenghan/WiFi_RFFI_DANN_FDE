import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from models import RFFI
import h5py as h5
from dataloader import DatasetBase


model_save_path = 'models/classifier0+1+3+4_l2norm_SGD_1000sample.pth'
transform = 'cfo_fde_40cbw'
rffi = RFFI(8, l2_norm=True).cuda()
rffi.load_state_dict(torch.load(model_save_path))
#model = nn.Sequential(*list(rffi.children())[:-1], *list(list(rffi.children())[-1].children())[:-1])

model = rffi.backbone
model.eval()


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


f = h5.File('dg_dataset.h5')
h5_group1 = f['5']


dataset1 = DatasetBase(h5_group1, transform)
train_set, test_set = __balance_val_split(dataset1, 0.995)
train_loader = DataLoader(train_set, batch_size=256, shuffle=False)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

def extract_features(data_loader, model):
    features = []
    labels = []
    with torch.no_grad():  
        for data, target in data_loader:
            output = model(data.cuda()).cpu()  
            features.append(output)
            labels.append(target)
    return torch.cat(features).numpy(), torch.cat(labels).numpy()


train_features, train_labels = extract_features(train_loader, model)


knn = KNeighborsClassifier(n_neighbors=3, metric='cosine')
knn.fit(train_features, train_labels)


test_features, test_labels = extract_features(test_loader, model)
predictions = knn.predict(test_features)


from sklearn.metrics import accuracy_score
print('Accuracy:', accuracy_score(test_labels, predictions))
