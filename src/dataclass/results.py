#pylint: skip-file

from dataclasses import dataclass

@dataclass
class Results:
    '''
    Results dataclass is used to store cuumulative results from the accuracy
    testing metrics during each test
    '''
    k: int
    average_precision: float = 0
    true_positives: int = 0
    false_positives: int = 0
    number_of_queries: int = 0
    number_of_answers: int = 0
    expected_number_of_answers: int = 0
    precision: float = 0

    def __post_init__(self):
        if self.k is None:
            raise ValueError("The 'k' parameter is required and cannot be None.")
        
    def update(self, true_pos: int, false_pos: int, num_answers: int, average_precision: float) -> None:
        self.average_precision += average_precision
        self.true_positives += true_pos
        self.false_positives += false_pos
        self.number_of_queries += 1
        self.number_of_answers += num_answers
        self.expected_number_of_answers += self.k
        self.precision = self.true_positives / self.expected_number_of_answers if self.expected_number_of_answers and self.true_positives > 0 else 0
    
    def get_map(self):
        return self.average_precision / self.number_of_queries if self.number_of_queries > 0 else 0
