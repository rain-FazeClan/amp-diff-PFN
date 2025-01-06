import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader

class MultiTaskGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_tasks):
        super(MultiTaskGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, hidden_channels)
        self.task_heads = torch.nn.ModuleList([torch.nn.Linear(hidden_channels, 2) for _ in range(num_tasks)])

    def forward(self, x, edge_index, batch):
        # Graph convolution layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Global mean pooling
        x = global_mean_pool(x, batch)

        # Fully connected layer
        x = self.fc(x)
        x = F.relu(x)

        # Task-specific heads
        outputs = [task_head(x) for task_head in self.task_heads]
        return outputs