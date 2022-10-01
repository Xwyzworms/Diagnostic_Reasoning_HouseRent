import pandas as pd
from pandas_profiling import ProfileReport
import sklearn
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from sklearn.preprocessing import  LabelEncoder



def getOnlyColmnsWithType(df, dataType):
	'''
	Referred from Jupyter Notebook.

	To select all numeric types, use np.number or 'number'

	To select strings you must use the object dtype but note that this will return all object dtype columns

	See the NumPy dtype hierarchy <http://docs.scipy.org/doc/numpy/reference/arrays.scalars.html>__

	To select datetimes, use np.datetime64, 'datetime' or 'datetime64'

	To select timedeltas, use np.timedelta64, 'timedelta' or 'timedelta64'

	To select Pandas categorical dtypes, use 'category'

	To select Pandas datetimetz dtypes, use 'datetimetz' (new in 0.20.0) or ``'datetime64[ns, tz]'
	'''
	return df.select_dtypes(include=[dataType]).columns

def profileReport(df):
	'''
	Referred from Jupyter Notebook.

	:param df: The dataframe to be analyzed
	'''
	profile = ProfileReport(df, title='Pandas Profiling Report', html={'style':{'full_width':True}})
	return profile

def normalizeNumericColumn(df,num_col):
	scaler = StandardScaler()
	df[num_col]=scaler.fit_transform(df[num_col])
	return df[num_col],scaler

def denormalizeNumericColumn(df,scaler,num_col):
	df[num_col]=scaler.inverse_transform(df[num_col])
	return df[num_col]


def encodeCategoricalColumn(df,cat_col=[]):
	if len(df.shape)==1:
		encoder = LabelEncoder()
		df = encoder.fit_transform(df)
		return df,encoder
	dictEncoder=defaultdict(LabelEncoder)
	df[cat_col] = df[cat_col].apply(lambda x: dictEncoder[x.name].fit_transform(x))
	return df, dictEncoder

def decodeCategoricalColumn(df,dictEncoder,cat_col):
	df[cat_col] = df[cat_col].apply(lambda x: dictEncoder[x.name].inverse_transform(x))
	return df

