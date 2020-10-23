def add_col_TEP_plus_1(df):
	df['TEP_plus_1'] = df['TEP_mean_uV_C'] + 1
	return df