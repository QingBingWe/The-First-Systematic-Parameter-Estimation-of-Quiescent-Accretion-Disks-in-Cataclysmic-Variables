import pandas as pd
sdss_file_name = '../experiments_new/lamost_similiest/lamost_finally_data.csv'  
lamost_file_name = '../experiments_new/sdss_similiest/sdss_finally_data.csv'
param_name = '../param_table.csv'
sdss_match_data = pd.read_csv(sdss_file_name)
lamost_match_data = pd.read_csv(lamost_file_name)
param_data = pd.read_csv(param_name)


all_data = []
for i in range(len(sdss_match_data)):
    single_data = {}
    single_data['Source'] = 'SDSS' 
    single_data['Name'] = sdss_match_data.iloc[i]['sdss_name']
    cloudy_name = sdss_match_data.iloc[i]['cloudy_name']
    index = int(cloudy_name.split('.')[0][4:])
    single_data['Blackbody Temperature (K)'] = param_data.iloc[index]['Teff']
    single_data['Luminosity (erg/s)'] = param_data.iloc[index]['logL']
    single_data['Inner Radius (lg(cm))'] = param_data.iloc[index]['Rin_log']
    single_data['Outer Radius (lg(cm))'] = param_data.iloc[index]['Rout_log']
    single_data['Hden (cm-3)'] = param_data.iloc[index]['hden_log']
    all_data.append(single_data)

for i in range(len(lamost_match_data)):
    single_data = {}
    single_data['Source'] = 'LAMOST' 
    single_data['Name'] = lamost_match_data.iloc[i]['sdss_name']
    cloudy_name = lamost_match_data.iloc[i]['cloudy_name']
    index = int(cloudy_name.split('.')[0][4:])
    single_data['Blackbody Temperature (K)'] = param_data.iloc[index]['Teff']
    single_data['Luminosity (erg/s)'] = param_data.iloc[index]['logL']
    single_data['Inner Radius (lg(cm))'] = param_data.iloc[index]['Rin_log']
    single_data['Outer Radius (lg(cm))'] = param_data.iloc[index]['Rout_log']
    single_data['Hden (cm-3)'] = param_data.iloc[index]['hden_log']
    all_data.append(single_data)
df = pd.DataFrame(all_data)
df.to_csv("../parameter_sdss_and_lamost.csv",index=False)
 