import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
from tqdm import tqdm, trange

# from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchvision.models import densenet121


# Hyperparameters
LEARNING_RATE = 0.007
BATCH_SIZE = 128
N_EPOCHS = 25
CLASSES = 7
IMAGE_HEIGHT = 48
IMAGE_WIDTH = 48
IMAGE_CHANNELS = 1

# Set random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Define the CNN model
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Define the Dataset class
class EmoDataset(Dataset):
    def __init__(self, csv_path, start_idx, end_idx, transform=None, is_test = False):
        self.dataset = pd.read_csv(csv_path).iloc[start_idx:end_idx]
        self.transform = transform
        self.is_test = is_test 

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image_id = self.dataset.iloc[index, 0] 
        image = self.dataset.iloc[index, 1]
        image = [int(pixel_value) for pixel_value in image.split(" ")]
        image = np.array(image).reshape(1, 48, 48)  # Reshape to (1, 48, 48)
        image = torch.from_numpy(image).float()  # Convert to PyTorch tensor and float

        # image = np.array(image).reshape(48, 48)  # Reshape to (48, 48)
        # # image = Image.fromarray(image)  # Convert to PIL Image
        # image = torch.from_numpy(image).float()

        if self.transform:
            image = self.transform(image)

        # image = image.float()

        if not self.is_test:  # For train/validation data
            label = int(self.dataset.iloc[index, 2])
            return image, label, image_id
        else:  # For test data, no label available
            return image, image_id

# Define transforms for data augmentation
transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10),

    # transforms.RandomResizedCrop((IMAGE_HEIGHT, IMAGE_WIDTH), scale=(0.8, 1.0)),  # Randomly resize and crop
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Apply color jitter
    # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Apply random affine transformations
    # transforms.ToTensor(),  # Convert to Tensor
    # transforms.Normalize(mean=[0.5], std=[0.5]), 
])

# Load datasets
train_set = EmoDataset(csv_path="dataset/train_dataset.csv", start_idx=0, end_idx=4500, transform=transform)
validation_set = EmoDataset(csv_path="dataset/train_dataset.csv", start_idx=4500, end_idx=5000)
test_set = EmoDataset(csv_path="dataset/test_dataset.csv", start_idx=0, end_idx=len(pd.read_csv("dataset/test_dataset.csv")), is_test=True)

train_loader = DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE)
val_loader = DataLoader(validation_set, shuffle=False, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_set, shuffle=False, batch_size=BATCH_SIZE)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# model = models.resnet18(weights = "DEFAULT")
# model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# model.fc = nn.Linear(model.fc.in_features, CLASSES)  # Adjust the final layer for our number of classes
# model = model.to(device)

# model = densenet121(weights = "DenseNet121_Weights.DEFAULT")  # Load a pre-trained DenseNet121 model
# model.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # Adjust for single channel input
# model.classifier = nn.Linear(model.classifier.in_features, CLASSES)  # Adjust final layer for number of classes
# model = model.to(device)

# from torchvision.models import vgg11

# model = vgg11(weights = 'VGG11_Weights.DEFAULT')  # Load a pre-trained VGG11 model
# model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # Adjust for single channel input
# model.classifier[6] = nn.Linear(4096, CLASSES)  # Adjust final layer for number of classes
# model = model.to(device)

# from torchvision.models import efficientnet_b0

# model = efficientnet_b0(weights= "EfficientNet_B0_Weights.DEFAULT")  # Load a pre-trained EfficientNet-B0 model
# model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)  # Adjust for single channel input
# model.classifier[1] = nn.Linear(model.classifier[1].in_features, CLASSES)  # Adjust final layer for number of classes
# model = model.to(device)


# Initialize the model, loss function, and optimizer
model = EmotionCNN(num_classes=CLASSES).to(device)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE,
                #  weight_decay=1e-4
                 )

scheduler = StepLR(optimizer, step_size=5, gamma=0.4)
# scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

criterion = nn.CrossEntropyLoss()

# Create a directory for saving models
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, "best_cnn_model.pth")  # Overwrite this file after each epoch

# Training loop
if __name__ == "__main__":
    best_val_accuracy = 0.0  # For tracking the best validation accuracy

    early_stop_patience = 10
    no_improvement_epochs = 0

    for epoch in trange(N_EPOCHS, desc="Training"):
        model.train()
        train_loss = 0.0
        correct, total = 0, 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y, ids = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)

        train_accuracy = correct / total * 100
        print(f"Epoch {epoch + 1}/{N_EPOCHS} training loss: {train_loss:.2f}, accuracy: {train_accuracy:.2f}%")

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1} in validation", leave=False):
                x, y, ids = batch
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                loss = criterion(y_hat, y)
                val_loss += loss.detach().cpu().item() / len(val_loader)

                correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
                total += len(x)

        val_accuracy = correct / total * 100
        print(f"Epoch {epoch + 1}/{N_EPOCHS} validation loss: {val_loss:.2f}, accuracy: {val_accuracy:.2f}%")

        # Save the model if the validation accuracy improves
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
            no_improvement_epochs = 0 
        else:
            no_improvement_epochs += 1
        
        # if no_improvement_epochs >= early_stop_patience:
        #     print("Early stopping triggered.")
        #     break

        scheduler.step()
        
    # Test loop
    model.eval()
    predictions = []
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            # x, y, ids = batch
            # x, y = x.to(device), y.to(device)
            # y_hat = model(x)
            # loss = criterion(y_hat, y)
            # test_loss += loss.detach().cpu().item() / len(test_loader)

            # correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            # total += len(x)
            x, ids = batch  # No label for test set
            x = x.to(device)
            y_hat = model(x)

            predicted_labels = torch.argmax(y_hat, dim=1).cpu().numpy()
            for image_id, predicted_label in zip(ids, predicted_labels):
                predictions.append((image_id, predicted_label))

        # print(f"Test loss: {test_loss:.2f}")
        # print(f"Test accuracy: {correct / total * 100:.2f}%")

    predictions_df = pd.DataFrame(predictions, columns=['id', 'label'])
    predictions_df.to_csv('test_predictions_self_cnn.csv', index=False)
    print("Predictions saved to test_predictions.csv")