import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import random

# Define the transformations for the datasets
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# CNN Model Template
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training function
def train_model(model, dataloader_train, dataloader_test, epochs, optimizer, scheduler, criterion, model_save_path):
    best_accuracy = 0.0
    for epoch in range(epochs):
        model.train()
        for images, labels in dataloader_train:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Evaluation on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in dataloader_test:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{epochs}], Accuracy: {accuracy:.2f}%')

        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), model_save_path)

    return best_accuracy

# Function to visualize sample images
def visualize_samples(data_dir, num_samples=5):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    class_names = dataset.classes
    
    plt.figure(figsize=(12, 8))
    for i in range(num_samples):
        img, label = dataset[random.randint(0, len(dataset) - 1)]
        plt.subplot(1, num_samples, i+1)
        plt.imshow(img.permute(1, 2, 0))
        plt.title(class_names[label])
        plt.axis('off')
    
    plt.show()

# Main script
if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Part 1: Dataset Preparation
    # Task 1 Train loader
    task1_train_dir = './data/task1/train'
    task1_test_dir = './data/task1/test'
    task1_dataset_train = datasets.ImageFolder(root=task1_train_dir, transform=transform)
    task1_dataset_test = datasets.ImageFolder(root=task1_test_dir, transform=transform)

    task1_dataloader_train = DataLoader(task1_dataset_train, batch_size=32, shuffle=True, num_workers=4)
    task1_dataloader_test = DataLoader(task1_dataset_test, batch_size=32, shuffle=False, num_workers=4)

    # Print class names and indices
    print("Task 1: Class names:", task1_dataset_train.classes)
    print("Task 1: Class to index mapping:", task1_dataset_train.class_to_idx)

    # Task 2 Train loader
    task2_train_dir = './data/task2/train'
    task2_test_dir = './data/task2/test'
    task2_dataset_train = datasets.ImageFolder(root=task2_train_dir, transform=transform)
    task2_dataset_test = datasets.ImageFolder(root=task2_test_dir, transform=transform)

    task2_dataloader_train = DataLoader(task2_dataset_train, batch_size=16, shuffle=True, num_workers=4)
    task2_dataloader_test = DataLoader(task2_dataset_test, batch_size=32, shuffle=False, num_workers=4)

    # Print class names and indices
    print("Task 2: Class names:", task2_dataset_train.classes)
    print("Task 2: Class to index mapping:", task2_dataset_train.class_to_idx)


    # Visualize Task 1 samples
    print("Task 1 Dataset Samples")
    visualize_samples('./data/task1/train')

    # Visualize Task 2 samples
    print("Task 2 Dataset Samples")
    visualize_samples('./data/task2/train')


    # Data for plotting
    models = ['A1 (Task 1)', 'A2 (Task 2)', 'A3 (Transfer Learning)']
    accuracies = [55.40, 39.40, 48.20]

    plt.figure(figsize=(10, 6))
    plt.bar(models, accuracies, color=['blue', 'red', 'green'])
    plt.xlabel('Models')
    plt.ylabel('Accuracy (%)')
    plt.title('Comparison of A1, A2, and A3 Accuracy')
    plt.ylim(0, 60)
    plt.show()


    '''
    # Part 3: Train and Evaluate model on Task 1
    model1 = CNNModel(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model1.parameters(), lr=0.1)
    scheduler = StepLR(optimizer, step_size=6, gamma=0.1)

    print("Training model1 on Task 1")
    A1_accuracy = train_model(model1, task1_dataloader_train, task1_dataloader_test, 20, optimizer, scheduler, criterion, 'best_model1.pth')
    print(f"A1 Accuracy: {A1_accuracy:.2f}%")

    # Part 4: Train and Evaluate model on Task 2
    model2 = CNNModel(num_classes=5).to(device)
    optimizer = optim.SGD(model2.parameters(), lr=0.1)
    scheduler = StepLR(optimizer, step_size=6, gamma=0.1)

    print("Training model2 on Task 2")
    A2_accuracy = train_model(model2, task2_dataloader_train, task2_dataloader_test, 20, optimizer, scheduler, criterion, 'best_model2.pth')
    print(f"A2 Accuracy: {A2_accuracy:.2f}%")

    # Part 5: Transfer Learning from Task 1 to Task 2
    model3 = CNNModel(num_classes=10).to(device)
    model3.load_state_dict(torch.load('best_model1.pth'))
    model3.fc1 = nn.Linear(64 * 16 * 16, 128) #Reinitializing the Second Last FC Layer
    model3.fc2 = nn.Linear(128, 5) #Replacing the Last FC Layer

    # Transfer learning with only last two layers trainable
    optimizer = optim.SGD([{'params': model3.fc1.parameters()}, {'params': model3.fc2.parameters()}], lr=0.01)
    scheduler = StepLR(optimizer, step_size=6, gamma=0.1)

    print("Training model3 (Transfer Learning) on Task 2")
    A3_accuracy = train_model(model3, task2_dataloader_train, task2_dataloader_test, 20, optimizer, scheduler, criterion, 'best_model3.pth')
    print(f"A3 Accuracy: {A3_accuracy:.2f}%")

    # Analysis
    print(f"Comparison of Accuracies:\nA1 (Task 1): {A1_accuracy:.2f}%\nA2 (Task 2): {A2_accuracy:.2f}%\nA3 (Transfer Learning Task 2): {A3_accuracy:.2f}%")

'''