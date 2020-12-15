

def extract_consumption_values(data_df):
    
    selected_columns = [col for col in data_df.columns if ":" in col]

    data_df_general = data_df[data_df['Consumption Category'] == 'GC']
    
    data_df_solar = data_df[data_df['Consumption Category'] == 'GG']
    
    return data_df_general[selected_columns], data_df_solar[selected_columns]
