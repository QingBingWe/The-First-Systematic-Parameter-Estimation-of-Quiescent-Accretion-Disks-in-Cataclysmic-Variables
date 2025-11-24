import pandas as pd

datapath = '../experiments_new/wgangp_SDSS/SDSS_wgangp_cloudy.csv'
data = pd.read_csv(datapath)
print(data.describe())