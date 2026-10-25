
import numpy as np


class Adaline:

    def __init__(self,learning_rate=0.01,epochs=100,use_bias=True,random_state=42,mse_threshold=0.01):
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.use_bias=use_bias
        self.rng=np.random.default_rng(random_state)
        self.weights=self.rng.uniform(-0.01,0.01,2)
        self.mse_threshold=mse_threshold
        self.bias=self.rng.uniform(-0.01,0.01)if use_bias else 0
        self.msePerEpoch=[]

    def predict(self,X):
        X=np.asarray(X,dtype=float)
        output=np.dot(X,self.weights)+self.bias
        return np.where(output>=0,1,-1)

    def fit(self,X,y):
        X=np.asarray(X,dtype=float)
        y=np.asarray(y,dtype=float)
        self.msePerEpoch=[]
        for epoch in range(self.epochs):
            """Shuffle the sample indices ashan makonsh kol epoch bashofhom be 
               nafs el sequence  """
            indices=self.rng.permutation(len(X))# btreturn array
            for i in indices:
                output=np.dot(X[i],self.weights)+self.bias
                error=y[i]-output
                old_weights=self.weights.copy()
                self.weights=(old_weights+self.learning_rate*error*X[i])
                if self.use_bias:
                    self.bias=self.bias+self.learning_rate*error*1
            # b3d ma kol epoch tkhlas: Calculate outputs using the updated weights
            outputs=np.dot(X,self.weights)+self.bias  
            errors=y-outputs
            mse=np.mean(errors**2)
            self.msePerEpoch.append(float(mse))#basave el mse bta3et el epoch de
            if mse<self.mse_threshold:
                break
        return self
