import pandas as pd
from pandas_profiling import ProfileReport

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