#pylint: skip-file

from dataclasses import dataclass
from typing import Optional

@dataclass
class Database:
    '''
    Database dataclass is used to store a reference to the database client
    (and collection if needed) being used in the current test.
    '''
    name: str
    client: object
    collection: Optional[object] = None

    def __post_init__(self):
        if self.client is None:
            raise ValueError("The 'client' parameter is required and cannot be None.")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("The 'name' parameter must be a non-empty string.")
