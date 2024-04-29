import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from Model import ResNet, BasicBlock
import matplotlib.pyplot as plt


def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)
    return train_loader, test_loader


def test_model(model, data_loader, sample_size=1000):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            if total >= sample_size:
                break
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    test_loss /= total
    accuracy = 100. * correct / total
    return 100 - accuracy  # Error percentage


def train_and_test_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=3, log_interval=100):
    train_errors = []
    test_errors = []

    total_iters = 0
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Log status every 'log_interval' iterations and at the end of each epoch
            if batch_idx % log_interval == 0 or batch_idx == len(train_loader) - 1:
                train_error = test_model(model, train_loader)
                test_error = test_model(model, test_loader)
                train_errors.append(train_error)
                test_errors.append(test_error)
                print(
                    f'Epoch {epoch + 1}/{num_epochs}, Iteration {total_iters}, Train Error: {train_error:.2f}%, Test Error: {test_error:.2f}%')
            total_iters += 1

    return train_errors, test_errors


def run_experiment(use_shortcut):
    train_loader, test_loader = load_data()
    configurations = {
        20: [2, 2, 2, 3],
        # 32: [3, 4, 4, 4],
        44: [5, 5, 5, 6],
        # 56: [6, 7, 7, 7],
        110: [14, 14, 13, 13],
    }
    results = {}
    for depth, blocks in configurations.items():
        model_name = f'ResNet{depth}' + ('' if use_shortcut else ' Plain')
        print(model_name)
        model = ResNet(BasicBlock, blocks, num_classes=10, use_shortcut=use_shortcut).cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_errors, test_errors = train_and_test_model(model, train_loader, test_loader, nn.CrossEntropyLoss(),
                                                         optimizer, num_epochs=80)
        results[model_name] = (train_errors, test_errors)
    return results


def plot_results(results, title):
    plt.figure(figsize=(12, 7))
    colors = ['r', 'g', 'b', 'y', 'm']
    for (key, (train_errors, test_errors)), color in zip(results.items(), colors):
        iters = range(len(train_errors))
        plt.plot(iters, test_errors, label=f'{key} Test', color=color, linestyle='-')
        plt.plot(iters, train_errors, label=f'{key} Train', color=color, linestyle='--')
    plt.xlabel('Iterations')
    plt.ylabel('Error %')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(title.replace(" ", "_") + '.png')
    # plt.show()


def main():
    resnet_results = run_experiment(True)
    plainnet_results = run_experiment(False)
    plot_results(resnet_results, 'ResNet Results')
    plot_results(plainnet_results, 'PlainNet Results')


if __name__ == "__main__":
    main()
