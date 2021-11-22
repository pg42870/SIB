

class LineRegressionReg:

    def __init__(self):


    def cost(self, X=None, y=None, theta = None):
        X =  add_itersect(X) if X is not None else self.X
        y = y if y is not None else self.y
        theta = theta if theta is not None else self.theta

        y_pred = np.dot(X,theta)
        return mse(y, y_pred)/2