from assignment_4.MultWeights import MultiplicativeWeights

class MultiplicativeWeights1(MultiplicativeWeights):

    def __init__(self):
        super().__init__(beta = 0.5)


    def update_weights(self, preds, outcome):
        #TODO
