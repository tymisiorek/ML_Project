import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.metrics import classification_report, confusion_matrix

seed = 123
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

graph_path = '/home/zem4by/ML/knn_subset_graph.pt'
output_path = '/home/zem4by/ML'

#params
epochs = 100 
hidden_dim = 64
dropout_rate = 0.5
lr = 0.01
weight_decay = 5e-4


data_dict = torch.load(graph_path, weights_only=False)
x_np = data_dict['X_scaled']
y_np = data_dict['y']
edge_index = data_dict['edge_index']
train_idx = data_dict['train_idx']
test_idx = data_dict['test_idx']

num_nodes, num_feats = x_np.shape
num_classes = int(y_np.max()) + 1

data = Data(x=torch.tensor(x_np, dtype=torch.float), edge_index=edge_index,y=torch.tensor(y_np, dtype=torch.long))
#make a train/test mask to prevent leakage when training
data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.train_mask[train_idx] = True
data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.test_mask[test_idx] = True


#use inverse frequency weights to prevent overly large weights on minority class
labels_train = data.y[data.train_mask].cpu().numpy()
counts = np.bincount(labels_train, minlength=num_classes)
inv = 1.0 / (counts + 1e-12)
w = np.sqrt(inv)
w = w / w.sum() * num_classes
weights = torch.tensor(w, dtype=torch.float) 

class GAT(torch.nn.Module):
    #2 layers
    #4 heads in first layer, single head second
    #elu
    def __init__(self, in_dim, hid_dim, out_dim, dropout=dropout_rate):
        super().__init__()
        self.conv1 = GATConv(in_dim, hid_dim, heads=4, dropout=dropout)
        self.conv2 = GATConv(hid_dim * 4, out_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, data):
        x, edge = data.x, data.edge_index
        x = self.conv1(x, edge)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge)
        return F.log_softmax(x, dim=1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = GAT(num_feats, hidden_dim, num_classes).to(device)
optimize = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
data = data.to(device) 
weights = weights.to(device)

#train model and keep performance metrics to make graphs
stats = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
for epoch in range(1, epochs + 1):
    model.train()
    optimize.zero_grad()
    out = model(data)
    mask_train = data.train_mask
    loss = F.nll_loss(out[mask_train], data.y[mask_train], weight=weights)
    loss.backward()
    optimize.step()

    model.eval()
    with torch.no_grad():
        out = model(data)
        train_loss = F.nll_loss(out[mask_train], data.y[mask_train], weight=weights).item()
        train_acc = (out[mask_train].argmax(1) == data.y[mask_train]).float().mean().item()
        mask_test = data.test_mask
        # include class weights here to match training loss computation
        test_loss = F.nll_loss(out[mask_test], data.y[mask_test], weight=weights).item()
        test_acc = (out[mask_test].argmax(1) == data.y[mask_test]).float().mean().item()

    stats['train_loss'].append(train_loss)
    stats['train_acc'].append(train_acc)
    stats['test_loss'].append(test_loss)
    stats['test_acc'].append(test_acc)
    print(f"Ep {epoch:03d} | L {train_loss:.3f} A {train_acc:.3f} | vL {test_loss:.3f} vA {test_acc:.3f}")


output_path.mkdir(exist_ok=True, parents=True)

#plot + save everything
plt.figure()
plt.plot(stats['train_loss'], label='train')
plt.plot(stats['test_loss'], label='test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig(output_path / 'loss_curve.png')

plt.figure()
plt.plot(stats['train_acc'], label='train')
plt.plot(stats['test_acc'], label='test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig(output_path / 'accuracy_curve.png')

y_pred = out[data.test_mask].argmax(1).cpu().tolist()
y_true = data.y[data.test_mask].cpu().tolist()

report_df = pd.DataFrame(classification_report(
    y_true,
    y_pred,
    labels=list(range(num_classes)),
    target_names=[f"class_{i}" for i in range(num_classes)],
    output_dict=True,
    zero_division=0
)).transpose()

report_df.to_csv(f"{output_path}/gcn_results.csv")