import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.models import resnet18
from tqdm import tqdm

# ============================================================
# DEVICE
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# SINKHORN RANK (DIFFERENTIABLE)
# ============================================================
def normalize_and_squash(x):
    mean = x.mean(dim=1, keepdim=True)
    xc = x - mean
    std = torch.norm(xc, dim=1, keepdim=True) / (x.shape[1] ** 0.5)
    std = torch.clamp(std, min=1e-6)
    xs = xc / std
    return (torch.atan(xs) + torch.pi / 2) / torch.pi


def sinkhorn_rank(x, epsilon=1e-3, max_iter=50):
    """
    x: (B, L)
    returns soft ranks (B, L)
    """
    B, L = x.shape
    a = torch.ones(L, device=x.device) / L
    b = torch.ones(L, device=x.device) / L
    y = torch.linspace(0, 1, L, device=x.device)

    x_tilde = normalize_and_squash(x)

    C = (x_tilde.unsqueeze(2) - y.view(1, 1, -1)) ** 2
    K = torch.exp(-C / epsilon)

    u = torch.ones(B, L, device=x.device)
    v = torch.ones(B, L, device=x.device)

    for _ in range(max_iter):
        v = b / (torch.matmul(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + 1e-8)
        u = a / (torch.matmul(K, v.unsqueeze(-1)).squeeze(-1) + 1e-8)

    P = u.unsqueeze(2) * K * v.unsqueeze(1)
    b_bar = torch.cumsum(b, dim=0)

    ranks = L * torch.matmul(P, b_bar)
    return ranks


# ============================================================
# SOFT TOP-1 LOSS (J_k with k=1)
# ============================================================
class SoftTop1Loss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits, labels):
        ranks = sinkhorn_rank(logits, self.epsilon)
        idx = torch.arange(logits.size(0), device=logits.device)
        r_true = ranks[idx, labels]
        loss = torch.relu(logits.size(1) - r_true)
        return loss.mean()


# ============================================================
# MODELS
# ============================================================
class VanillaCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        return self.fc(x)


# ============================================================
# DATA
# ============================================================
def get_loaders(dataset="cifar10", batch_size=128):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    if dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        num_classes = 10
    else:
        trainset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
        num_classes = 100

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    return trainloader, testloader, num_classes


# ============================================================
# TRAIN / EVAL
# ============================================================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_acc(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


# ============================================================
# EXPERIMENT
# ============================================================
def run(model_name="cnn", dataset="cifar10", epochs=200, soft=True):
    trainloader, testloader, C = get_loaders(dataset)
    if model_name == "cnn":
        model = VanillaCNN(C)
    else:
        model = resnet18(num_classes=C)

    model.to(device)

    if soft:
        criterion = SoftTop1Loss(epsilon=1e-3)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    accs = []
    for e in range(epochs):
        train_epoch(model, trainloader, optimizer, criterion)
        acc = eval_acc(model, testloader)
        accs.append(acc)
        print(f"[{model_name} | {dataset} | {'soft' if soft else 'CE'}] Epoch {e+1} Acc={acc:.4f}")

    return accs


# ============================================================
# MAIN -> FIGURE 4
# ============================================================
if __name__ == "__main__":
    epochs = 600

    # Figure 4-1
    acc_cnn_ce = run("cnn", "cifar10", epochs, soft=False)
    acc_cnn_soft = run("cnn", "cifar10", epochs, soft=True)

    plt.plot(acc_cnn_ce, label="CNN Cross-Entropy")
    plt.plot(acc_cnn_soft, label="CNN Soft Error")
    plt.legend()
    plt.title("Figure 4-1 - CNN CIFAR-10")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

    # Figure 4-2
    acc_resnet_ce = run("resnet", "cifar10", epochs, soft=False)
    acc_resnet_soft = run("resnet", "cifar10", epochs, soft=True)

    plt.figure()
    plt.plot(acc_resnet_ce, label="ResNet18 Cross-Entropy")
    plt.plot(acc_resnet_soft, label="ResNet18 Soft Error")
    plt.legend()
    plt.title("Figure 4-2 - ResNet18 CIFAR-10")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()