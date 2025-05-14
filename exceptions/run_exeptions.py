from typing import Text, Optional
from typeguard import typechecked


@typechecked
class RunException(Exception):
    def __init__(
        self, *args, message: Optional[Text] = "Error occurred when attempting to run."
    ):
        self.message = message
        super().__init__(*args, self.message)
