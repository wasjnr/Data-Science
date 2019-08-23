#sys.path.append('models')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
#conda install -c conda-forge scikit-plot
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from scipy.io import arff

class MainPipeline:
   
    data = arff.loadarff('total_casos.arff')
    v110 = pd.DataFrame(data[0])
       
    latest = v110
   
    training_columns = ['pop_residente', 'taxa_alfabetizacao', 'den_demografica', 'rendimento_medio']
    target_column = 'taxa_alfabetizacao'

    def _gerar_modelo_completo(self, dataframe=None, X_columns=None, y_column=None):
        if dataframe is None:
            dataframe = self.carrega_csv()

        if X_columns is None:
            X_columns = self.training_columns

        if y_column is None:
            y_column = self.target_column
            
        selected_columns = X_columns + [y_column]

        print("Colunas: ")
        print(dataframe.columns)
        print("---\n\nInfo: ")
        print(dataframe[selected_columns].info())
        print("---\n\nHeatmap: ")
        self.printHeatmap(dataframe[selected_columns])

        self.gera_modelo_random_forest(dataframe, X_columns, y_column)
        return dataframe;


    #def getDescricaoColunas():
     #   return cols

    def printHeatmap(self, dataframe):
        corr = dataframe.corr()
        sns.heatmap(corr)

    def carrega_csv(self, filename = None):
        if filename is None:
            filename = self.latest
        path = 'C:\\Users\\was\\Desktop\\Data\\' + filename
        return pd.read_csv(path)
        #return pd.read_csv(path, index_col='Unnamed: 0')

    def salvar_csv(dataframe, filename):
        path = 'C:\\Users\\was\\Desktop\\Data\\'+filename+'.csv'
        dataframe.to_csv(path, index=False)

    def gera_modelo_random_forest(self, dataframe, X_columns, y_column):
        X = dataframe[X_columns]
        y = dataframe[y_column]

        model = RandomForestClassifier()
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilidades = model.predict_proba(X_test)
        print("---\n\nAccuracy: ")
        self.calcula_accuracy(y_test, predictions)
        print("---\n\nClassification Report: ")
        self.calcula_classification_report(y_test, predictions)
        print("-")
        self.gera_roc(y_test, probabilidades)
        print("-")
        self.gera_ks(y_test, probabilidades)
        print("-")
        self.gera_feature_importances(model, X_columns)
        print("-")
        return model
    
    def gera_modelo_arvore(self, dataframe, X_columns, y_column):
        X = dataframe[X_columns]
        y = dataframe[y_column]

        model = DecisionTreeClassifier()
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilidades = model.predict_proba(X_test)
        print("---\n\nAccuracy: ")
        self.calcula_accuracy(y_test, predictions)
        print("---\n\nClassification Report: ")
        self.calcula_classification_report(y_test, predictions)
        print("-")
        self.gera_roc(y_test, probabilidades)
        print("-")
        self.gera_kfold(X, y, dataframe)
        print("-")
        self.gera_ks(y_test, probabilidades)
        print("-")
        self.gera_feature_importances(model, X_columns)
        print("-")
        return model
    
    def gera_kfold(self, X, y, df):
        kf = KFold(n_splits=10)
        kf.get_n_splits(X)
        fold = 0
        folds_list = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model = DecisionTreeClassifier()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)*100
            current_fold = {
                "Fold": fold,
                "Accuracy": accuracy,
                "Train size": len(X_train),
                "Test size": len(X_test),
                "% train size": len(X_train)*100/len(df),
                "% test size": len(X_test)*100/len(df)
            }
            fold = fold + 1
            folds_list.append(current_fold)

        dataframe_resultado = pd.DataFrame(folds_list)[["Fold", "Accuracy", "Train size","Test size","% train size","% test size"]]
        print("K-Fold (10 folds) \n\tAcurácia média: ", dataframe_resultado["Accuracy"].mean, "%")
        return 
    
    def gera_roc(self, y_test, prob):
        skplt.metrics.plot_roc_curve(y_test, prob)
        
    def gera_ks(self, y_test, prob):
        skplt.metrics.plot_ks_statistic(y_test, prob)
        
    def gera_feature_importances(self, model, column_names):
        skplt.estimators.plot_feature_importances(model, feature_names=column_names, x_tick_rotation=90)

    def calcula_accuracy(self, y_test, predictions):
        print("\tacc=", accuracy_score(y_test, predictions)*100, "%")

    def calcula_classification_report(self, y_test, predictions):
        print(classification_report(y_test, predictions))

    def categoriza_dataframe(dataframe, coluna, a, b, c, d):
        if(
            a != None and
            b != None and
            c != None and
            d != None
        ):
            dataframe[coluna] = dataframe[coluna].apply(_bin_5_categorias(self, a, b, c, d))
            return

        if(
            a != None and
            b != None and
            c != None and
            d is None
        ):
            dataframe[coluna] = dataframe[coluna].apply(_bin_4_categorias(self, a, b, c))
            return

        if(
            a != None and
            b != None and
            c is None and
            d is None
        ):
            dataframe[coluna] = dataframe[coluna].apply(_bin_3_categorias(self, a, b))
            return

        if (
            a != None and
            b is None and
            c is None and
            d is None
        ):
            dataframe[coluna] = dataframe[coluna].apply(_bin_2_categorias(self, a))
            return

    def _bin_5_categorias(valor, a, b, c, d):
        if valor <= a:
            return 1
        elif valor > a and valor <= b:
            return 2
        elif valor > b and valor <= c:
            return 3
        elif valor > c and valor <= d:
            return 4
        else:
            return 5

    def _bin_4_categorias(valor, a, b, c):
        if valor <= a:
            return 1
        elif valor > a and valor <= b:
            return 2
        elif valor > b and valor <= c:
            return 3
        else:
            return 4

    def _bin_3_categorias(valor, a, b):
        if valor <= a:
            return 1
        elif valor > a and valor <= b:
            return 2
        else:
            return 3

    def _bin_2_categorias(valor, a):
        if valor <= a:
            return 1
        else:
            return 2
