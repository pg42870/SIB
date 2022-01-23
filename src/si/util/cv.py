from ..util.util import train_test_split, add_intersect
import numpy as np
import itertools

all = ['CrossValidationScore', 'GridSearchCV']

class CrossValidationScore:

    def __init__(self, model, dataset, score=None, **kwargs):
        self.model = model
        self.dataset = dataset
        self.cv = kwargs.get('cv', 3)
        self.score = score
        self.split = kwargs.get('split', 0.8)
        #self.score = kwargs.get('score', None)
        self.train_scores = None
        self.test_scores = None
        self.ds = None
    """
    def run(self):
        train_scores = []
        test_scores = []
        ds = [] #save the datasets
        for _ in range(self.cv): # _ pois nao precisamos do valor da iteracao
            train, test = train_test_split(self.dataset, self.split)
            ds.append((train, test))
            self.model.fit(train)
            train_scores.append(self.model.cost())
            test_scores.append(self.model.cost(test.X, test.Y))
        self.train_scores = train_scores
        self.test_scores = test_scores
        self.ds = ds
        return train_scores, test_scores
    """
    def run(self):
        train_scores = []
        test_scores = []
        ds = []
        for _ in range(self.cv):
            train, test = train_test_split(self.dataset, self.split)
            ds.append((train, test))
            self.model.fit(train)
            if not self.score:
                train_scores.append(self.model.cost())
                test_scores.append(self.model.cost(test.X, test.Y))
            else:
                y_train = np.ma.apply_along_axis(self.model.predict, axis=1, arr=train.X)
                train_scores.append(self.score(train.Y, y_train))
                y_test = np.ma.apply_along_axis(self.model.predict, axis=1, arr=test.X)
                test_scores.append(self.score(test.Y, y_test))
        self.train_scores = train_scores
        self.test_scores = test_scores
        self.ds = ds
        return train_scores, test_scores


    def toDataframe(self):
        import pandas as pd
        assert self.train_scores and self.test_scores, "Need to run first"
        return pd.DataFrame({'Train Scores': self.train_scores, 'Test Scores': self.test_scores})

class GridSearchCV:

    def __init__(self, model, dataset, parameters, score=None, **kwargs):
        self.model = model
        self.dataset = dataset
        self.score = score
        hasparam = [hasattr(self.model, param) for param in parameters] #verify if keys are or not atributes
        if np.all(hasparam):
            self.parameters = parameters
        else:
            index = hasparam.index(False)
            keys = list(parameters.keys())
            raise ValueError(f" Wrong parameters: {keys[index]}")
        self.kwargs = kwargs
        self.results = None

    def run(self):
        self.results = []
        attrs = list(self.parameters.keys())
        values = list(self.parameters.values())
        from itertools import product
        for comb in list(product(*values)):
            for attr, value in zip(attrs, comb):
                setattr(self.model, attr, value)
            cv = CrossValidationScore(self.model, self.dataset, self.score, **self.kwargs)
            cv.run()
            self.results.append(cv.run())
        return self.results

    #def toDataframe(self):
        #import panda as pd
        #assert self.results, "The grid search needs to be ran."
        #data = dict()
        #for i, k in enumerate(self.parameters.keys()):
            #v = []
            #for r in self.results:
               # v.append(r[0][i])
            #data[k] = v
            #ate aqui temos as duas primeiras colunas criadas faltam as outras



    def toDataframe(self):
        import pandas as pd
        assert self.results, 'The grid search needs to be ran first'
        data = dict()
        for i, k in enumerate(self.parameters.keys()):
            v = list()
            for r in self.results:
                v.append(r[0][i])
            data[k] = v
        for i in range(len(self.results[0][1][0])):
            train, test = list(), list()
            for res in self.results:
                train.append(res[1][0][i])
                test.append(res[1][1][i])
            data['Train'+str(i+1)] = train
            data['Test'+str(i+1)] = test
            return pd.DataFrame(data)