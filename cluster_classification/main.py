import os

import torch
from torch import nn

from cluster_classification.model import Model
from cluster_classification.test import test_loop
from cluster_classification.train import train_loop
from cluster_classification.data import get_data, DatasourceType
from cluster_classification.classification_logger import CleverLogger

logger = CleverLogger(__name__)

# After how many epochs should a checkpoint be made?
CHECKPOINT_RATE = 10
# If accuracy growth falls below this value, stop training early
GROWTH_THRESHOLD = 0.001
# If accuracy reaches this this threshold, then stop training.
STOP_THRESHOLD = 0.95
# Learning rate of the model.
LEARNING_RATE = 1e-4
# How many data points to analyze in a batch.
BATCH_SIZE = 1

# Instantiate a Model object:
model = Model()

# Set device
current_device = torch.accelerator.current_accelerator(check_available=True)
if current_device is not None:
    device = current_device
else:
    device = torch.device("cpu")

logger.log_notice(f"Using {device.type} device")

# Set loss function
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

# Collect data
logger.log_start_major_process('load_data')
training_data = get_data(DatasourceType.TRAINING, BATCH_SIZE)
testing_data = get_data(DatasourceType.TESTING, BATCH_SIZE)
logger.log_end_major_process('load_data')

# Stochastic Gradient Descent
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

last_acc = 0.0
sentinal = True
epoch = 0
# Epoch loop
logger.log_start_major_process('train_test_loop')
while sentinal:
    logger.log_notice(f"Epoch {epoch}")
    logger.log_start_minor_process("training")
    # Run through the training data once
    train_loop(device, training_data, model, loss_fn, optimizer)
    logger.log_end_minor_process('training')

    logger.log_start_minor_process("testing")
    # Run through the testing data once and evaluate the model's accuracy
    acc, string = test_loop(device, testing_data, model, loss_fn)
    logger.log_end_minor_process('testing')

    # If the epoch is a checpoint epoch,
    if epoch % CHECKPOINT_RATE == 0:
        logger.log_notice(f"Checkpoint {epoch // CHECKPOINT_RATE}")
        logger.log_notice(string)
        # save the model as a checkpoint
        torch.save(
            model, f"checkpoint-{(epoch // CHECKPOINT_RATE):>05d}-classification.pth"
        )
        if acc > STOP_THRESHOLD:
            # If the accuracy is hight enough, exit training
            logger.log_notice(f"Accuracy threshold reached: {acc}")
            sentinal = False
        if acc - last_acc < GROWTH_THRESHOLD:
            # If the accuacy has not grown appreciably since last test, exit
            # training
            logger.log_notice(f"Accuracy growth limit reached: {acc - last_acc}")
            sentinal = False
        last_acc = acc
    epoch = epoch + 1
logger.log_end_major_process('train_test_loop')

# Softlink the last checkpoint
os.symlink(
    f"checkpoint-{(epoch // CHECKPOINT_RATE):>05d}-classification.pth",
    "current-classification.pth",
)
