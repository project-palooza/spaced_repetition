from spaced_repetition import *

def leitner_model():
    """
    leitner requires 2 features:
    history_correct
    history_incorrect
    """
    leitner = SpacedRepetition(2,0,0)
    hard_coded_leitner_params = torch.tensor([[1.0],[-1]], dtype=torch.float32)
    leitner.theta.data = hard_coded_leitner_params

    return leitner