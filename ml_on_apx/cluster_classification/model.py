from torch import nn

from ml_on_apx.cleverlogger import CleverLogger


class Model(nn.Module):
    """The structure of the classification model.

    Extends: `torch.nn.Module`
    """

    def __init__(self):
        super().__init__()
        self.logger = CleverLogger(__name__)
        self.logger.log_enter_function("model_constructor")
        # Other layers to try: Dropout and batch normalization,
        # if they make any sense. It's a small model though, so likely not
        self.stack = nn.Sequential(
            # Output based on what was provided by the CCD
            nn.Linear(18, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            # em similarity, tau similarity
        )
        self.logger.log_exit_function("model_constructor")

    def forward(self, x):
        self.logger.log_enter_function("model_forward")
        certainties = self.stack(x)
        self.logger.log_function_exit_type("return", retval=certainties)
        self.logger.log_exit_function("model_forward")
        return certainties
