# Machine Learning Testbed Platform - Incremental User Stories

1. As a user, I want to be able to train and test a simple fully connected neural network against the MNIST dataset.: Completed

2. As a user, I want to see a progress bar for each epoch during training. After each iteration, the progress bar should disappear and be replaced with a single line like: {epoch:>3} - ({train_time:>6.2f},{test_time:>6.2f}) - training accuracy {train_accuracy:>5.2f} ({best_accuracy:>5.2f}) - training loss {train_loss:>6.4f} ({best_loss:>6.4f}) - test accuracy {test_accuracy:>5.2f} ({best_test_accuracy:>5.2f}) - test loss {test_loss:>6.4f} ({best_test_loss:>6.4f}). Each epoch should be numbered.: Completed

3. As a user, I want to be able to configure whether to use CPU or GPU for training and testing.: Completed

4. As a developer, I want to add configurable data transformations (ToTensor, Normalize, Resize, etc.) to the data loading pipeline, so that I can easily experiment with different preprocessing techniques.: Completed (with unit tests)

5. As a user, I want to be able to configure the training script using a single configuration file (e.g., YAML or JSON). This file should allow me to specify parameters such as the model architecture, dataset, transformations, optimizer, learning rate, batch size, number of epochs, and device (CPU or GPU). This will make it easier to manage and reproduce experiments.: Completed

6. As a developer, I want to implement a caching mechanism for the transformed training and testing samples in the datasets. This should prevent redundant recalculation of transformations for each epoch.: Completed

7. As a user, I want to be able to configure transformations with their parameters. For example resize with the size for each channel or normalize with the parameters.: Completed

8. As a user, I want to be able to configure different datasets, for example MNIST, CIFAR10, CIFAR100 or others.: Completed

9. As a user, I want to be able to configure different losses (MSELoss, MAESLoss, CrossEntropyLoss, etc).: Completed (with refactoring and tests)

10. As a user, I want to be able to configure multiple different metrics to track. For example accuracy, f1, confusion matrix, precision, recall, AUC, ROC, and average precision.: Completed

11. As a user, I want to be able to configure multiple optimizers for the training, such as SGD, Adam, and RMSprop, with their parameters.: Not Started

12. As a user, I want the application to use wandb to track all the metrics and results, and the run name should include major parameters from the configuration (e.g., prefix, model, optimizer, timestamp) for easy identification.: Not Started
