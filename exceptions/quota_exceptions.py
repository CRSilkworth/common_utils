from typing import Text, Optional
from typeguard import typechecked


@typechecked
class QuotaException(Exception):
    def __init__(self, *args, message: Optional[Text] = "Quota Exceeded"):
        self.message = message
        super().__init__(*args, self.message)
