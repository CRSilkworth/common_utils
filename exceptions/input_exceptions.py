from typing import Text, Optional
from typeguard import typechecked


@typechecked
class InputException(Exception):
    def __init__(self, *args, message: Optional[Text] = "Parsing input failed."):
        self.message = message
        super().__init__(*args, self.message)


@typechecked
class SetAttributeException(ValueError):
    def __init__(self, *args, message: Optional[Text] = "Failed to set attribute."):
        self.message = message
        super().__init__(*args, self.message)
