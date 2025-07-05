'''
IMPORTANT NOTE:
    - Transformers excel at temporal/positional order of data
        - Text, stock prices, sensor data, pose estimation, speech/audio, biological sequences
    - Good for large datasets
    - Model expects input shape: [batch_size, sequence_length, input_dim]
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import math
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
from preprocess_files import getXY, printLabelInfo, normalize_path, make_output_dirs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EyeGazeTransformer(nn.Module):
    def __init__(self, num_classes, input_dim=7, d_model = 64, nhead = 4, num_layers = 2, max_seq_length = 1250):
        super().__init__()

        #1. Project input_dimensions to d_model (Use when our input feature size doesn't match d_model)
        self.embedding_layer = nn.Linear(input_dim, d_model)

        #2. Positional encoding adds position info to each step so model knows where in sequence each one is(Good when order of input matters)
        self.pos_encoding_layer = PositionalEncoding(d_model, max_seq_length)

        #3. Encoder layer makes the model understand patterns and relationships between all parts of input sequence(How does step 8 affect step 99?)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model = d_model, nhead = nhead, dropout = 0.1, batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers = num_layers) # Creates 2 instances of the encoder layer

        self.dropout = nn.Dropout(p=0.1)  # Drop 10% of neurons randomly

        #4. Classification head(layer) turns output of the transformer into actual human readable outputs(scores for each class)
        self.classification_layer = nn.Linear(d_model, num_classes)

        #5. Forward method tells the model how the input travels through all previous layers
    def forward(self, x):
        x = self.embedding_layer(x) # Shape: (batch_size, window_length, d_model)
        x = self.dropout(x)  # Dropout after embedding
        x = self.pos_encoding_layer(x) # Same shape: (batch_size, window_length, d_model)
        x = self.transformer_encoder(x) # Still: (batch_size, window_length, d_model)
        x = self.dropout(x)  # Optional: Dropout again after transformer

        mean = x.mean(dim=1) # Combines all output feature vectors from each time step and averages them into one single vector

        x = self.classification_layer(mean) # Shape: (batch_size, num_classes)

        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class GazeDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = x_data
        self.y = y_data

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        x_tensor = torch.tensor(self.x[index], dtype=torch.float32)
        y_tensor = torch.tensor(int(self.y[index]), dtype=torch.long)
        return x_tensor, y_tensor # X-Shape: [batch_size, window_size(rows), columns_in_window(num_of_features) or [windows, rows, columns ]]

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
    model = EyeGazeTransformer(num_classes = num_classes).to(device)
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
    model = model.to(device)
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