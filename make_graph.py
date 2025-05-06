import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors

data_path = '/home/zem4by/ML/full_data.csv'
output_path = '/home/zem4by/ML/knn_subset_graph.pt'

fraction = .2 
random_state = 42
k = 6 
test_size = 0.20 

df = pd.read_csv(data_path)
df = df.sample(frac=fraction, random_state=random_state).reset_index(drop=True)
# print(df.shape)
#columns to remove
id_cols = [
    'Flow ID','Source IP','Destination IP',
    'Source Port','Destination Port','Protocol','Timestamp'
]
df_feats = df.drop(columns=id_cols)

#get rid of missing values and drop the label so we dont cheat the prediction
X_num = (
    df_feats
    .drop(columns=['Label'])
    .select_dtypes(include=[np.number])
    .replace([np.inf, -np.inf], np.nan)
)
valid = X_num.notna().all(axis=1)
X_num = X_num.loc[valid].reset_index(drop=True)
y_raw = df_feats['Label'].loc[valid].reset_index(drop=True)

#encode labels
label_encoder = LabelEncoder().fit(y_raw)
y_encoded = label_encoder.transform(y_raw)

#stratify sample
sampled = StratifiedShuffleSplit(
    n_splits=1, test_size=test_size, random_state=random_state
)
train_idx, test_idx = next(sampled.split(X_num, y_encoded))

#standardize
scaler = StandardScaler().fit(X_num.iloc[train_idx])
X_scaled = scaler.transform(X_num).astype(np.float32)

#create knn graph
kneighbors = NearestNeighbors(n_neighbors=k+1, n_jobs=-1).fit(X_scaled)
indices = kneighbors.kneighbors(X_scaled, return_distance=False)[:, 1:]
source = np.repeat(np.arange(X_scaled.shape[0], dtype=np.int64), k)
dest = indices.ravel()

is_training = np.zeros(X_scaled.shape[0], dtype=bool)
is_test = np.zeros_like(is_training)
is_training[train_idx] = True
is_test[test_idx]  = True

#only keep edges where both ends are in the same split
mask_keep = (is_training[source] & is_training[dest]) | (is_test[source] & is_test[dest])
src, dst = source[mask_keep], dest[mask_keep]
# print(mask_keep)
src_sym = np.concatenate([src, dst])
dst_sym = np.concatenate([dst, src])

edge_index = torch.tensor(np.vstack([src_sym, dst_sym]), dtype=torch.long)

os.makedirs(os.path.dirname(output_path), exist_ok=True)
torch.save({
    'X_scaled'  : X_scaled,
    'y'         : y_encoded,
    'train_idx' : train_idx.astype(np.int64),
    'test_idx'  : test_idx.astype(np.int64),
    'edge_index': edge_index,
}, output_path)

print("finished building graph")
