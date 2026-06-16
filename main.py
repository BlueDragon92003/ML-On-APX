def main():
    """
    Parse args:
    - manage_datasets
    - manage_models
    - train model

    Manage datasets:
      Load dataset index
      Show list of datasets
      - Create
      - Read
      - Update
      - Delete

    Manage models:
      Load model index
      Show list of models & checkpoints
      - Save checkpoints
      - Delete models
      - Test models
      - Define model
        - Datasets & labels
        - Inputs
        - Hyperparemeters
      - Create model training job
        - Stop conditions (n epochs, min accuracy, min accuracy growth rate)

    Train model:
      Fork for each job
      Load relavant information
      Train until stop conditions

    """

    return 0


if __name__ != "__main__":
    main()
