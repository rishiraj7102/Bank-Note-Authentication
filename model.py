import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import pickle
df=pd.read_csv("BankNote_Authentication.csv")
df.head()
X=df.iloc[:,:-1]
Y=df.iloc[:,-1:]
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(X,Y)
from sklearn.model_selection import cross_val_score
cross_val_score(model,X,Y,cv=5).mean()
filename="bankauthorization.pkl"
pickle.dump(model,open(filename,'wb'))