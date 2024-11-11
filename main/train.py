import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from models import EnhancedSparseAttention

class ImageClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Initial convolution layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Sparse attention layer
        self.attention = EnhancedSparseAttention(
            in_channels=128,
            out_channels=128,
            num_heads=4,
            attention_ratio=0.5,
            sparsity_pattern=SparsityPattern.AXIAL,
            use_relative_pos=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.attention(x)
        x = self.classifier(x)
        return x

# 2. Training utilities
class Trainer:
    def __init__(self, model, device, save_path='model.pth'):
        self.model = model.to(device)
        self.device = device
        self.save_path = save_path
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=3, factor=0.5
        )
        
        self.train_losses = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}: Loss {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, val_loader):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = 100. * correct / total
        self.val_accuracies.append(accuracy)
        return accuracy
    
    def train(self, train_loader, val_loader, epochs=10):
        best_acc = 0
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            
            # Training
            loss = self.train_epoch(train_loader)
            print(f'Average training loss: {loss:.4f}')
            
            # Validation
            accuracy = self.validate(val_loader)
            print(f'Validation accuracy: {accuracy:.2f}%')
            
            # Learning rate scheduling
            self.scheduler.step(accuracy)
            
            # Save best model
            if accuracy > best_acc:
                best_acc = accuracy
                torch.save(self.model.state_dict(), self.save_path)
                print(f'Model saved with accuracy {accuracy:.2f}%')
        
        return self.train_losses, self.val_accuracies

# 3. Visualization utilities
def plot_attention_maps(model, sample_input):
    """Plot attention maps for each head."""
    # Get attention maps
    with torch.no_grad():
        _ = model(sample_input)
    
    attention_maps = next(iter(model.attention.attention_maps.values()))
    num_heads = attention_maps.size(1)
    
    fig, axes = plt.subplots(2, num_heads//2, figsize=(15, 6))
    axes = axes.flatten()
    
    for head in range(num_heads):
        attn_map = attention_maps[0, head].cpu().numpy()
        im = axes[head].imshow(attn_map, cmap='viridis')
        axes[head].set_title(f'Head {head+1}')
        plt.colorbar(im, ax=axes[head])
    
    plt.tight_layout()
    return fig

def plot_training_history(trainer):
    """Plot training loss and validation accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training loss
    ax1.plot(trainer.train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    
    # Plot validation accuracy
    ax2.plot(trainer.val_accuracies)
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    
    plt.tight_layout()
    return fig

# 4. Main training script
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    # Create model and trainer
    model = ImageClassifier(num_classes=10)
    trainer = Trainer(model, device)
    
    # Train the model
    train_losses, val_accuracies = trainer.train(
        train_loader, test_loader, epochs=10
    )
    
    # Plot training history
    history_fig = plot_training_history(trainer)
    history_fig.savefig('training_history.png')
    
    # Plot attention maps
    sample_input = next(iter(test_loader))[0][:1].to(device)
    attention_fig = plot_attention_maps(model, sample_input)
    attention_fig.savefig('attention_maps.png')
    
    print("Training completed! Check training_history.png and attention_maps.png")

if __name__ == '__main__':
    main()