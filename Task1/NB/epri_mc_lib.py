
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

###################### General functions ######################

def add_col_TEP_plus_1(df):
	df['TEP_plus_1'] = df['TEP_mean_uV_C'] + 1
	return df

# Scale a dataframe using a given scaler (not yet fit)
# Keeps index and column names
# Return new dataframe, scaler
def scale_general(df, scaler):
    df_scaled = pd.DataFrame(scaler.fit_transform(df))
    df_scaled.columns = df.columns
    df_scaled.index = df.index
    return df_scaled, scaler

# Scale a data frame using MinMaxScaler
# Keeps index and column names
# returns new data frame
def scale_min_max(df):
    return scale_general(df, MinMaxScaler())

# Scale a data frame using StandardScaler
# Keeps index and column names
# returns new data frame
def scale_standard_scaler(df):
    return scale_general(df, StandardScaler())

# Transform a dataframe using an existing scaler
# Keeps index and column names
# returns new data frame
def transform_df(scaler, df):
    df_scaled = pd.DataFrame(scaler.transform(df))
    df_scaled.columns = df.columns
    df_scaled.index = df.index
    return df_scaled

# Get lower and upper error bounds of a measurement
# inputs: df- pandas data frame, 
#         measures_list - list of column names with measured values
#         errors_list - list of column names with error values (stdev or error, etc)
# returns two dataframes, df_lower_boundary measures with errors subtracted, 
#          df_upper_boundary measures with error added
def calc_error_bounds(df, measures_list, errors_list):
    df_measures = df[measures_list]
    df_errors = df[errors_list]
    df_lower_boundary = df_measures - df_errors.values
    df_upper_boundary = df_measures + df_errors.values
    return df_lower_boundary, df_upper_boundary



###################### Lists for handling dataframes ######################

drop_list_absorption_500_200 = ['Absorption_avg_500','Absorption_std_500','Absorption_avg_200','Absorption_std_200']
drop_list_absorption_100 = ['Absorption_avg_100','Absorption_std_100']
