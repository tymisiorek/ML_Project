import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

graph_path = '/home/zem4by/ML/knn_subset_graph.pt'
output_path = '/home/zem4by/ML'

seed = 123
epochs = 100
hidden_dim = 64
dropout_p = 0.5
lr = 0.01
weight_decay = 5e-4

torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

d = torch.load(graph_path, weights_only=False)
x_np = d['X_scaled']
y_np = d['y']
edge_index = d['edge_index']
train_idx = d['train_idx']
test_idx = d['test_idx']

node_count, feature_count = x_np.shape
num_classes = int(y_np.max()) + 1

data = Data(x=torch.tensor(x_np, dtype=torch.float), edge_index=edge_index, y=torch.tensor(y_np, dtype=torch.long))
#make a train/test mask to prevent leakage when training
data.train_mask = torch.zeros(node_count, dtype=torch.bool); data.train_mask[train_idx] = True
data.test_mask = torch.zeros(node_count, dtype=torch.bool); data.test_mask[test_idx]  = True

#use inv frequency class weighting
counts = np.bincount(y_np[train_idx], minlength=num_classes)
weight = np.sqrt(1. / (counts + 1e-12))
weight = weight / weight.sum() * num_classes
weights = torch.tensor(weight, dtype=torch.float)

class GCN(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim, cached=False, normalize=True)
        self.conv2 = GCNConv(hid_dim, out_dim, cached=False, normalize=True)
        self.dropout = dropout

    def forward(self, data):
        x, edge = data.x, data.edge_index
        x = self.conv1(x, edge)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge)
        return F.log_softmax(x, dim=1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = GCN(feature_count, hidden_dim, num_classes, dropout_p).to(device)
optimize = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
data = data.to(device)
weights = weights.to(device)

#train and keep stats for graphs
stats = {'train_loss':[], 'test_loss':[], 'train_acc':[], 'test_acc':[]}
for epoch in range(1, epochs + 1):
    model.train()
    optimize.zero_grad()
    out = model(data)
    mask = data.train_mask
    loss = F.nll_loss(out[mask], data.y[mask], weight=weights)
    loss.backward()
    optimize.step()

    model.eval()
    with torch.no_grad():
        out = model(data)
        training_loss = F.nll_loss(out[mask], data.y[mask], weight=weights).item()
        training_acc = (out[mask].argmax(1) == data.y[mask]).float().mean().item()
        test_mask = data.test_mask
        test_loss= F.nll_loss(out[test_mask], data.y[test_mask], weight=weights).item()
        test_acc = (out[test_mask].argmax(1) == data.y[test_mask]).float().mean().item()

    stats['train_loss'].append(training_loss)
    stats['test_loss'].append(test_loss)
    stats['train_acc'].append(training_acc)
    stats['test_acc'].append(test_acc)
    print(f"Ep {epoch:03d} | L {training_loss:.3f} A {training_acc:.3f} | vL {test_loss:.3f} vA {test_acc:.3f}")

output_path.mkdir(exist_ok=True, parents=True)

#plot everything
plt.figure(); plt.plot(stats['train_loss'], label='train'); plt.plot(stats['test_loss'], label='test')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.tight_layout()
plt.savefig(output_path / 'loss_curve.png')

plt.figure(); plt.plot(stats['train_acc'], label='train'); plt.plot(stats['test_acc'], label='test')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.tight_layout()
plt.savefig(output_path / 'accuracy_curve.png')

test_mask_np = data.test_mask.cpu().numpy()
y_true = data.y[test_mask_np].cpu().tolist()
y_pred = model(data).argmax(1).cpu().numpy()[test_mask_np].tolist()

class_ids   = sorted(set(y_true))
class_names = [f"class {i}" for i in class_ids]

report_df = pd.DataFrame(classification_report(
    y_true, y_pred,
    labels=class_ids,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)).transpose()
report_df.to_csv(f"{output_path}\gcn_results.csv")

torch.save(model.state_dict(), output_path / 'gcn_intrusion_model.pt')
print("done")
