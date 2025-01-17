# libraries: machine learning/AI
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms, models


from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, classification_report, f1_score,recall_score,precision_score
# libraries: plotting
import seaborn as sns
import matplotlib.pyplot as plt
# libraries: utilities
import numpy as np
import random
import pandas as pd
import datetime as dt
from collections import Counter

# from playsound import playsound
# Ensure reproducibility
# Fix random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # Set to any fixed value for reproducibility

# paths
path = 'C:\\Users\\NZ\\.cache\\kagglehub\\datasets\\paultimothymooney\\chest-xray-pneumonia\\versions\\2\\chest_xray\\'
folder_val = path+'val\\'
folder_test = path+'test\\'
folder_train = path+'train\\'
class_labels = ['NORMAL','PNEUMONIA']
image_size = 150
image_size = 224
# Task 1: Load the images

# Data augmentation for training data
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure grayscale input
    transforms.RandomHorizontalFlip(p=0.5),       # Horizontal flip
    transforms.RandomRotation(degrees=10),        # Random rotation ±10°
    transforms.RandomResizedCrop(size=224, scale=(0.9, 1.1)),  # Random zoom
    transforms.ColorJitter(brightness=0.1, contrast=0.1),      # Brightness/contrast
    transforms.ToTensor(),                        # Convert to Tensor
    transforms.Normalize(mean=[0.5], std=[0.5])   # Normalize pixel values
])

# Validation/Test data transformation (no augmentation)
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure grayscale input
    transforms.Resize((224, 224)),                # Resize to match input size
    transforms.ToTensor(),                        # Convert to Tensor
    transforms.Normalize(mean=[0.5], std=[0.5])   # Normalize pixel values
])

# Load datasets
train_dataset = ImageFolder(root=folder_train, transform=train_transform)
val_dataset = ImageFolder(root=folder_val, transform=test_transform)
test_dataset = ImageFolder(root=folder_test, transform=test_transform)

# Task 2: (5 marks) ---------------------------------------
# Train and compare three Machine Learning/Deep Learning Models
class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(input_size, 2)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)

class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 75 * 75, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
    
log_reg_model = LogisticRegression(input_size=image_size * image_size)
# cnn_model = SmallCNN()

# Load the pretrained VGG16 model
vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
# Modify the first convolutional layer to accept grayscale input
new_conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
new_conv1.weight.data = vgg16.features[0].weight.data.mean(dim=1, keepdim=True)
vgg16.features[0] = new_conv1
# Replace the final classifier layer for binary classification
vgg16.classifier[6] = nn.Linear(4096, 2)

# Define ResNet-18 model
resnet = models.resnet18(pretrained=False)
resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Adjust for single-channel input
resnet.fc = nn.Linear(resnet.fc.in_features, 2)  # Binary classification


def execute_model(model,results,model_name='',learningrate=0.001,epochs=10,batch_size=64):
    # global results
    global class_labels
    global train_dataset
    global val_dataset
    global test_dataset
    
    start = dt.datetime.now()
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    print(model_name, ' - ', device_name)
    model = model.to(device)
    
    set_seed(42)
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learningrate)
    
    
    train_loss, val_losses = [], []
    train_acc, val_accs = [], []
    # Train the model
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.squeeze().long().to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Track loss, accuracy, and predictions for AUC
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs[:, 1].detach().cpu().numpy())  # Use probabilities for class 1
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        epoch_auc = roc_auc_score(all_labels, all_preds)
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        # print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
        # print(f"Train Accuracy: {epoch_acc:.4f}, Train AUC: {epoch_auc:.4f}")
    
        # Validate the model
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_labels = []
        val_preds = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                # inputs, labels = inputs.to(device), labels.to(device)
                inputs, labels = inputs.to(device), labels.squeeze().long().to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(outputs[:, 1].cpu().numpy())
        
        val_acc = val_correct / val_total
        val_auc = roc_auc_score(val_labels, val_preds)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        # print(f"Validation Accuracy: {val_acc:.4f}, Validation AUC: {val_auc:.4f}")
    
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_losses,label='val_loss')
    plt.title(model_name)
    plt.legend()
    plt.show()
    
    plt.plot(train_acc, label='train_acc')
    plt.plot(val_accs,label='val_acc')
    plt.title(model_name)
    plt.legend()
    plt.show()

    if(model_name=='VGG-16'):
        model_file_path='vgg16_trained_model_2.pth'
        # Save the trained model
        print(f"Saving the trained model to {model_file_path}")
        torch.save(model.state_dict(), model_file_path)
    
    # Test the model
    model.eval()
    test_labels = []
    test_preds = []
    test_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.squeeze().long().to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            test_labels.extend(labels.cpu().numpy())
            test_preds.extend(predicted.cpu().numpy())
            test_probs.extend(outputs[:, 1].cpu().numpy())  # Use probabilities for class 1
    
    # Calculate test metrics
    test_accuracy = accuracy_score(test_labels, test_preds)
    test_auc = roc_auc_score(test_labels, test_probs)
    conf_matrix = confusion_matrix(test_labels, test_preds)
    
    print(f"Test Accuracy: {test_accuracy:.4f}, Test AUC: {test_auc:.4f}")
    
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # Classification report for precision, recall, and F1-score
    class_report = classification_report(test_labels, test_preds, target_names=class_labels, digits=4)
    
    print("Classification Report - " + model_name + ":")
    print(class_report)
    
    # Plot confusion matrix
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - ' + model_name)
    plt.show()
    
    classes = list(set(test_labels))
    precision = precision_score(test_labels, test_preds, average=None, labels=classes)
    recall = recall_score(test_labels, test_preds, average=None, labels=classes)
    f1 = f1_score(test_labels, test_preds, average=None, labels=classes)
    # Count the number of data points in each class
    class_counts = Counter(test_labels)
    
    end = dt.datetime.now()
    duration = end - start
    
    class_i = 0
    results['Precision-'+class_labels[class_i]].append(precision[class_i])
    results['Recall-'+class_labels[class_i]].append(recall[class_i])
    results['F1-Score-'+class_labels[class_i]].append(f1[class_i])
    results['Support-'+class_labels[class_i]].append(class_counts[class_i])
    
    class_i = 1 #
    results['Precision-'+class_labels[class_i]].append(precision[class_i])
    results['Recall-'+class_labels[class_i]].append(recall[class_i])
    results['F1-Score-'+class_labels[class_i]].append(f1[class_i])
    results['Support-'+class_labels[class_i]].append(class_counts[class_i])
    
    
    results['ModelName'].append(model_name)
    results['Epochs'].append(epochs)
    results['BatchSize'].append(batch_size)
    results['Device'].append(device_name)
    results['LearningRate'].append(learningrate)
    results['Start'].append(start.strftime("H%M%S"))
    results['Duration (minutes)'].append(duration.total_seconds()/60)
    
    print('Duration (minutes):',duration.total_seconds()/60)
    
    return results
    
results = {'ModelName':[]
            ,'Epochs':[]
            ,'BatchSize':[]
            ,'Device':[]
            ,'LearningRate':[]
            ,'Start':[]
            ,'Duration (minutes)':[]
            ,'Precision-'+class_labels[0]:[]
            ,'Recall-'+class_labels[0]:[]
            ,'F1-Score-'+class_labels[0]:[]
            ,'Support-'+class_labels[0]:[]
            ,'Precision-'+class_labels[1]:[]
            ,'Recall-'+class_labels[1]:[]
            ,'F1-Score-'+class_labels[1]:[]
            ,'Support-'+class_labels[1]:[]
          }

results=execute_model(log_reg_model,results,'Logistic Regression',learningrate=0.0001)
# results=execute_model(cnn_model,results,'Simple CNN')
results=execute_model(resnet,results,'ResNet-18',learningrate=0.0001)
results=execute_model(vgg16,results,'VGG-16',learningrate=0.0001)

df = pd.DataFrame(results)
df.to_csv('results_'+dt.datetime.now().strftime("%m%d_%H%M")+'.csv', index=False)

# playsound('Sounds\\beep-01a.mp3')
