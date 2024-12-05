Part 1: Dataset Preparation (1 Mark)
- Download the Asn2.zip dataset and copy it to a folder named “data”
Download Link: https://drive.google.com/file/d/1woIbIj0VyiK0uORqb0QICAFrnztsepGP/view?usp=sharing 
Alternate Download Link: https://mega.nz/file/GGYVwQaK#9Gsq2UbyU02VeNmHUKlUib6PXMwoOaEPrmkoiTmUB5c
- Extract the zip file
- Visualize task 1 and task 2 train datasets.
- Load the data into dataloaders using the following code in a python file which should be placed in the parent folder of the “data” folder.

# Task 1 Train loader
data_dir = './data/task1/train'
task1_dataset_train = datasets.ImageFolder(root=data_dir, transform=transform)
task1_dataloader_train = DataLoader(task1_dataset_train, batch_size=32, shuffle=True, num_workers=4)


# Print class names and indices
print("Task 1: Class names:", task1_dataset_train.classes)
print("Task 1: Class to index mapping:", task1_dataset_train.class_to_idx)

# Task 1 Test loader
data_dir = './data/task1/test'
task1_dataset_test = datasets.ImageFolder(root=data_dir, transform=transform)
task1_dataloader_test = DataLoader(task1_dataset_test, batch_size=32, shuffle=False, num_workers=4)

# Task 2 Train loader
data_dir = './data/task2/train'
task2_dataset_train = datasets.ImageFolder(root=data_dir, transform=transform)
task1_dataloader_train = DataLoader(task2_dataset_train , batch_size=16, shuffle=True, num_workers=4)


# Print class names and indices
print("Task 2: Class names:", task2_dataset_train.classes)
print("Task 2: Class to index mapping:", task2_dataset_train.class_to_idx)

# Task 2 Test loader
data_dir = './data/task2/test'
task2_dataset_test = datasets.ImageFolder(root=data_dir, transform=transform)
task1_dataloader_test = DataLoader(task2_dataset_test , batch_size=32, shuffle=False, num_workers=4)



Part 2: CNN Architecture (2)

**CNN Model Template**:
- Construct a CNN model with the following architecture:
- **Input Layer**: Accepts input images.
- **Convolutional Layer 1**: 32 filters, 3x3 kernel size, ReLU activation.
- **Pooling Layer 1**: Max Pooling with a 2x2 window and stride=2

- **Convolutional Layer 2**: 64 filters, 3x3 kernel size, ReLU activation.
- **Pooling Layer 2**: Max Pooling with a 2x2 window and stride=2

- **Convolutional Layer 3**: 64 filters, 3x3 kernel size, ReLU activation.
- **Pooling Layer 3**: Max Pooling with a 2x2 window and stride=2

- **Fully Connected Layer**: 128 units, ReLU activation.
- **Output Layer**: Softmax activation for classification.


Part 3: Train and Evaluate model on Task 1 (2)
Create an object “model1” of the above CNN model template with output size matching the number of classes in Task1.
Use CrossEntropyLoss as the loss function and SGD as the optimizer with a learning rate 0.1 and a lr_scheduler.StepLR that multiplies the learning rate by a factor of 0.1 after every step of 6 epochs
Train the model1 on Task 1 train data for 20 epochs and evaluate it on Task 1 test data after every training epoch
After every epoch check if the current model1 achieves the highest test accuracy on the Task 1 test data and if so save it as best_model1.pth
Mention this accuracy in the report as A1 accuracy

Part 4: Train and Evaluate model on Task 2 (2)
Create an object “model2” of the above CNN model template with output size matching the number of classes in Task2.
Use CrossEntropyLoss as the loss function and SGD as the optimizer with a learning rate 0.1 and a lr_scheduler.StepLR that multiplies the learning rate by a factor of 0.1 after every step of 6 epochs
Train the model2 on Task 2 train data for 20 epochs and evaluate it on Task 2 test data after every training epoch
After every epoch check if the current model2 achieves the highest test accuracy on the Task 2 test data and if so save it as best_model2.pth
Mention this accuracy in the report as A2 accuracy

Part 5: Transfer learning from Task 1 to Task 2 (4)
Create a new object “model3” of the above CNN model template the best_model1.pth object “model2” of the above CNN model template with output size matching the number of classes in Task1.
Load the best_model1.pth into model3
Reinitialize the second last FC layer of model3.
Replace the last FC layer of model3 with another FC layer of output size matching the number of classes in Task2.
Use CrossEntropyLoss as the loss function
Use SGD as the optimizer defined as follows assuming the the last 2 fc layers are model3.fc1 and model3.fc2, such that only these 2 layers are trained
	optimizer = optim.SGD([{'params': model3.fc1.parameters()},{'params': model3.fc2.parameters()}], lr=0.01)

Use a lr_scheduler.StepLR that multiplies the learning rate by a factor of 0.1 after every step of 6 epochs
Train the model3 on Task 2 train data for 20 epochs and evaluate it on Task 2 test data after every training epoch
After every epoch check if the current model3 achieves the highest test accuracy on the Task 2 test data and if so save it as best_model3.pth
Mention this accuracy in the report as A3 accuracy.

Part 6: Analysis (3)
Compare the A1, A2, and A3 accuracy.
Explain the difference between the A2 and A3 accuracy, even though both models have the same architecture and are trained on the same dataset.
