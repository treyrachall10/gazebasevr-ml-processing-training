'''
IMPORTANT NOTE:
    - Good for spatial and temporal patterns
    - Good for small datasets
        - Especially when dropout is included
    - Model expects input shape: [batch_size, in_channels, sequence_length]
'''
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from argparse import ArgumentParser
from preprocess_files import getXY, printLabelInfo, normalize_path, make_output_dirs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class DenseNetModel(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        # 1. Creates dense block
        self.dense_layer = DenseBlock1D(in_channels)

        # 6. Creates global average pooling layer (takes features and compresses them into a single number)
        self.global_avg_layer = nn.AdaptiveAvgPool1d(1)

        #7. Creates linear embedding layer (turns input(feature map) to specified size)
        total_features = in_channels + 8 * 32
        self.embedding_layer = nn.Linear(total_features, 128)

        #7. Creates linear embedding layer (turns input(feature map) to specified size)
        self.classification_layer = nn.Linear(128, num_classes)

    # Tells model how input should flow
    def forward(self, x):
        x = self.dense_layer(x)
        x = self.global_avg_layer(x)
        x = x.squeeze(-1)
        x = self.embedding_layer(x)
        x = self.classification_layer(x)
        
        return x
        
        

class DenseBlock1D(nn.Module):
    def __init__(self, in_channels, growth_rate=32, num_layers=8):
        super().__init__()

        # 2. Creates 3 lists containing the different layers in block
        self.relu_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()
        self.conv1_layers = nn.ModuleList()

        for i in range(num_layers):
            if i != 0:
                # 3. Creates the batch normalization layer (normalizes inputs to a layer)
                batch_norm_layer = nn.BatchNorm1d(in_channels + i * growth_rate)
                batch_norm_layer.weight.data.fill_(1.0)
                batch_norm_layer.bias.data.zero_()

                # 4. Creates ReLU activation layer (sets all negative values to 0, keeps positives unchanged; adds non-linearity)
                relu_layer = nn.ReLU()
                self.batch_norm_layers.append(batch_norm_layer)
                self.relu_layers.append(relu_layer)

            dilation = 2 ** (i % 7)
            padding = dilation

                # 5. Creates convolutional 1D layer (extracts local temporal features by sliding filters across the time dimension)
            conv = nn.Conv1d(
                in_channels + i * growth_rate,
                growth_rate,
                kernel_size=3,
                stride=1,
                padding=padding,
                dilation=dilation,
                bias=False
            )
            self.conv1_layers.append(conv)

    # Tells model how input should flow
    def forward(self, x):
        features = [x]
        for i in range(len(self.conv1_layers)):
                
            if i == 0:
                x = self.conv1_layers[0](x)
            else:
                x = torch.cat(features, dim = 1)
                x = self.batch_norm_layers[i - 1](x)
                x = self.relu_layers[i - 1](x)
                x = self.conv1_layers[i](x)

            features.append(x)

        return torch.cat(features, dim = 1)
    
class GazeDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = x_data
        self.y = y_data

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        x_tensor = torch.tensor(self.x[index], dtype=torch.float32).transpose(0, 1)  # Transpose from [1250, 7] to [7, 1250]
        y_tensor = torch.tensor(int(self.y[index]), dtype=torch.long)
        return x_tensor, y_tensor
    
# Calculates difference between prediction and true label
def lossFunction(predictions, labels):
    criterion = nn.CrossEntropyLoss() # Creates instance of CrossEntropyLoss class
    loss = criterion(predictions, labels) # Calls Forward() in CrossEntropyLoss class - Predictions shape: [batch_size, num_classes] Labels shape: [batch_size]
    return loss # Just one number

# Training loop
def trainOneEpoch(epoch_index, tb_writer, training_loader, optimizer, model):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero gradients for every batch
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = lossFunction(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

# Validates accuracy of model at time of calling
def validate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.4f}")
    model.train()

# Loop for one epoch
def trainingLoop(tb_writer, training_loader, validation_loader, optimizer, model):
   for epoch in range(50):
        print(f"Epoch {epoch + 1}/{50}")
        loss = trainOneEpoch(epoch, tb_writer, training_loader, optimizer, model)
        print(f"Epoch {epoch + 1} Loss: {loss}") 
        validate(model, validation_loader)

# Returns the full model initialized with input channels and total number of classes
def getModel(y):
    num_classes = len(set(y))
    model = DenseNetModel(7, num_classes).to(device)
    print("Returning model")
    return model

# Splits the dataset into training and testing sets (80/20 split)
def getTrainingSplit(x, y):
    dataset = GazeDataset(x, y)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    print("Returning data loaders")
    return train_dataset, test_dataset

# Initializes and returns the Adam optimizer for model training (controls how models weights are updated during training to minimize loss)
def getOptimizer(model):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("Returning optimizer")
    return optimizer

# Creates and returns a TensorBoard writer to log training metrics
def getTbWriter():
    tb_writer = SummaryWriter()
    print("Returning summary writer")
    return tb_writer

# Wraps datasets into DataLoader objects to feed data in batches to the model
def getLoaders(train_dataset, test_dataset):
    training_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # Feeds data into model - Turns input shape into [batch_size, window_rows, features]
    validation_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) # Tests the data
    print("Returning dataloaders")
    return training_loader, validation_loader

# Runs the full training process across all epochs, with validation each epoch
def mainLoop(x, y):
    model = getModel(y)
    train_dataset, test_dataset = getTrainingSplit(x, y)
    optimizer = getOptimizer(model)
    tb_writer = getTbWriter()
    training_loader, validation_loader = getLoaders(train_dataset, test_dataset)
    trainingLoop(tb_writer, training_loader, validation_loader, optimizer, model)

if __name__ == "__main__":
    parser = ArgumentParser(description = "Preprocess round 1 files from GazeBaseVR data set")
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Path to existing directory containing GazeBaseVR data",
    )
    parser.add_argument(
        "--round_1_dir",
        type=str,
        required=True,
        help="Path to output directory for storing round 1 files"
    )
    parser.add_argument(
        "--norm_dir",
        type=str,
        required=True,
        help="path to output directory for storing normalized data"
    )
    args = parser.parse_args()

    input_dir = normalize_path(args.src)
    round_1_dir = normalize_path(args.round_1_dir)
    norm_dir = normalize_path(args.norm_dir)

    make_output_dirs(round_1_dir)
    make_output_dirs(norm_dir)
    x, y = getXY(input_dir, round_1_dir, norm_dir)
    printLabelInfo(y)
    mainLoop(x, y)