from .model import Model
import numpy as np

def majority(values):
    return max(set(values), key=values.count)

def average(values):
    return sum(values)/len(values)




class Ensemble(Model):

    def __init__(self, models, f_vote, score): #modelos que nao estao treinados
        super().__init__()
        self.models= models
        self.fvote = f_vote
        self.score = score

    def fit(self, dataset): #vamos ter de treinar os modelos
        self.dataset = dataset
        for model in self.models:
            model.fit(dataset)
        self.is_fitted = True

    def predict(self, x):
        assert self.is_fitted, "Model must be fit before prediction"
        preds= [model.predict(x) for model in self.models]
        vote = self.fvote(preds)
        return vote

    def cost(self, X=None, y= None):
        X = X if X is not None else self.dataset.X
        y = y if y is not None else self.dataset.Y
        y_pred = np.ma.apply_along_axis(self.predict,
                                        axis=0, arr=X.T)
        return self.score(y, y_pred)