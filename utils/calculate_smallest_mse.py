import pandas as pd
import shutil
import os


def read_data(data_path):
    data = pd.read_csv(data_path)
    return data

os.makedirs('../experiments_new/sdss_similiest',exist_ok =True)
save_dir = '../experiments_new/sdss_similiest'
for i in ['real','normalized']:
    os.makedirs(os.path.join(save_dir,i),exist_ok=True)
finally_data = []
model_names = ['vae','wgangp','diffusion']
model_to_path = {
    'vae': '../experiments_new/vae_SDSS',
    'wgangp': '../experiments_new/wgangp_SDSS',
    'diffusion': '../experiments_new/diffusion_SDSS',
}
total_data = []
for key,value in model_to_path.items():
    total_data.append(read_data(os.path.join(value,f"SDSS_{key}_cloudy.csv")))

for i in total_data[0].index:
    each_finally_data = {}
    sdss_name = total_data[0].iloc[i]['sdss_file_name']
    each_finally_data['sdss_name'] = sdss_name
    samllest_mse=1000000
    small_model = ''
    small_cloudy_file_name = ''
    for j,data in enumerate(total_data):
        if data.iloc[i]['mse'] < samllest_mse:
            samllest_mse=data.iloc[i]['mse']
            small_model = model_names[j]
            small_cloudy_file_name = data.iloc[i]['cloudy_file_name']
            real_mse = data.iloc[i]['real_mse']
    each_finally_data['mse'] = samllest_mse
    each_finally_data['cloudy_name'] = small_cloudy_file_name
    each_finally_data['real_mse'] = real_mse
    shutil.copy(os.path.join(model_to_path[small_model],'real',sdss_name+'.png'),os.path.join(save_dir,'real',sdss_name+'.png'))
    shutil.copy(os.path.join(model_to_path[small_model],'normalized',sdss_name+'.png'),os.path.join(save_dir,'normalized',sdss_name+'.png'))

    finally_data.append(each_finally_data)
df = pd.DataFrame(finally_data)
df.to_csv(os.path.join(save_dir,'sdss_finally_data.csv'),index=False)
            



    
