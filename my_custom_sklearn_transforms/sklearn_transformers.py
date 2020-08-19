from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()

        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

class RenameColumns(BaseEstimator, TransformerMixin):
    """Classe para renomear as colunas do dataset"""
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
  
    def transform(self, X):
        return pd.DataFrame(X, columns=self.columns[2:])

class SumColumns(BaseEstimator, TransformerMixin):
    """Classe para somar valores em cada linha de determinadas colunas"""
    def __init__(self, columns, new_column_name):
        self.columns = columns
        self.new_column_name = new_column_name
  
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()
        data[self.new_column_name] = data[self.columns].sum(axis=1)

        return data

class MeanColumns(BaseEstimator, TransformerMixin):
    """Classe para tirar as médias dos valores em cada linha de determinadas colunas"""
    def __init__(self, columns, new_column_name):
        self.columns = columns
        self.new_column_name = new_column_name
  
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()
        data[self.new_column_name] = data[self.columns].mean(axis=1)

        return data

class CoefSum(BaseEstimator, TransformerMixin):
    """Classe destinada a realizar um cálculo de coeficiente. Neste caso, se soma as colunas de 
       horas de aulas com trabalhos concluídos e subtrai as faltas"""
    def __init__(self, columns, new_column_name):
        self.columns = columns
        self.new_column_name = new_column_name
  
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()
        data[self.new_column_name] = data[self.columns[0]] + data[self.columns[1]] - data[self.columns[2]]

        return data

class CoefMulti(BaseEstimator, TransformerMixin):
    """Classe destinada a realizar um cálculo de coeficiente. Neste caso, se multiplica as 
       colunas de horas de aulas com trabalhos concluídos e divide pelas faltas"""
    def __init__(self, columns, new_column_name):
        self.columns = columns
        self.new_column_name = new_column_name
  
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()
        data[self.new_column_name] = data[self.columns[0]] * data[self.columns[1]] / data[self.columns[2]]
        
        return data

class SimpleFillna(BaseEstimator, TransformerMixin):
    """Classe para substituir valores NAN por value"""
    def __init__(self, columns, value):
        self.columns = columns
        self.value = value
  
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()
        data[self.columns] = data[self.columns].fillna(self.value)

        return data
