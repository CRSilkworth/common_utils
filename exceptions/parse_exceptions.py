from typing import Text, Optional
from typeguard import typechecked


@typechecked
class ParseException(Exception):
    def __init__(
        self,
        *args,
        message: Optional[Text] = "Error occurred when attempting to parse."
    ):
        self.message = message
        super().__init__(*args, self.message)
