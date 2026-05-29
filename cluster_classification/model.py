from torch import nn

from cluster_classification.classification_logger import CleverLogger

logger = CleverLogger(__name__)

logger.log_start_load_module()

class Model(nn.Module):
    """The structure of the classification model.

    Extends: `torch.nn.Module`
    """

    def __init__(self):
        super().__init__()
        logger.log_enter_function('model_constructor')
        # Other layers to try: Dropout and batch normalization,
        # if they make any sense. It's a small model though, so likely not
        self.stack = nn.Sequential(
            # Eta, Phi, HCAL energies, HCAL deposition, Cluster Energy
            nn.Linear(2 + 9 + 9 + 1, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            # em similarity, tau similarity
        )
        logger.log_exit_function('model_constructor')

    def forward(self, x):
        logger.log_enter_function("model_forward")
        certainties = self.stack(x)
        logger.log_function_exit_type('return', retval=certainties)
        logger.log_exit_function("model_forward")
        return certainties


logger.log_end_load_module()
