from typing import Text, Optional
from typeguard import typechecked


@typechecked
class DependencyException(Exception):
    def __init__(self, *args, message: Optional[Text] = "Dependency not set"):
        self.message = message
        super().__init__(*args, self.message)
