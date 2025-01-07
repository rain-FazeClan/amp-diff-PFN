import torch
import pandas as pd
from torch_geometric.data import Data, DataLoader
import argparse
import os
import time
from model import MultiTaskGNN

def load_data(file_path):
    df = pd.read_csv(file_path)
    sequences = df['Sequence']
    labels = df.iloc[:, -1]
    features = df.iloc[:, 3:-1]

    # Ensure all features are numeric
    features = features.apply(pd.to_numeric, errors='coerce').fillna(0)

    data_list = []
    for i in range(len(df)):
        x = torch.tensor(features.iloc[i].values, dtype=torch.float).view(-1, 1)
        y = torch.tensor(labels[i], dtype=torch.long)
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)  # Dummy edge index
        batch = torch.tensor([i], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y, batch=batch)
        data_list.append(data)

    return data_list

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for data in loader:
        optimizer.zero_grad()
        outputs = model(data.x, data.edge_index, data.batch)
        loss = sum(criterion(output, data.y) for output in outputs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

def test(model, loader, criterion):
    model.eval()
    total_correct = 0
    total_loss = 0

    with torch.no_grad():
        for data in loader:
            outputs = model(data.x, data.edge_index, data.batch)
            loss = sum(criterion(output, data.y) for output in outputs)
            total_loss += loss.item()
            preds = [output.argmax(dim=1) for output in outputs]
            total_correct += sum((pred == data.y).sum().item() for pred in preds)

    accuracy = total_correct / (len(loader.dataset) * len(outputs))
    return total_loss / len(loader), accuracy

def main(args):
    # Record start time
    start_time = time.time()

    # Load data
    train_data = load_data(args.train_file)
    test_data = load_data(args.test_file)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # Initialize model, optimizer, and loss function
    num_node_features = train_data[0].x.size(1)
    hidden_channels = args.hidden_channels
    num_tasks = 5  # Assuming 5 tasks for 5 different bacteria

    model = MultiTaskGNN(num_node_features, hidden_channels, num_tasks)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    best_accuracy = 0.0

    # Training loop
    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, optimizer, criterion)
        test_loss, test_accuracy = test(model, test_loader, criterion)
        print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

        # Step the scheduler
        scheduler.step()

        # Save the best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            if not os.path.exists('models'):
                os.makedirs('models')
            torch.save(model.state_dict(), 'models/best_model.pth')

    # Record end time
    end_time = time.time()
    total_time = end_time - start_time
    print(f'Total training time: {total_time:.2f} seconds')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a multi-task GNN for antimicrobial peptide prediction.')
    parser.add_argument('--train_file', type=str, required=True, help='Path to the training data file')
    parser.add_argument('--test_file', type=str, required=True, help='Path to the testing data file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--hidden_channels', type=int, default=64, help='Number of hidden channels in GNN layers')

    args = parser.parse_args()
    main(args)