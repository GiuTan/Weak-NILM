import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 

file_agg_path = '/raid/users/eprincipi/clean_refit/dataset_weak/aggregate_data_noised/'
labels_path = '/raid/users/eprincipi/clean_refit/dataset_weak/labels/'
destination_agg_resample_path = '/raid/users/eprincipi/resampled_agg_REFIT_test/'
destination_labels_resample_path = '/raid/users/eprincipi/resampled_labels_REFIT_test/'
k = 4
quantity = 12000
for i in range(quantity):
    agg1 = np.load(file_agg_path + 'house_' + str(k) + '/aggregate_%d.npy' % i)
    labels_strong = np.load(labels_path + 'house_' + str(k) + '/strong_labels_%d.npy' % i, allow_pickle=True)
    
    

    time = pd.date_range('2014-01-01', periods=2550, freq='8s')
    arr = pd.Series(data=agg1,index=time)
    new_labels = []
    for a in range(5):
          label = pd.Series(data=labels_strong[a], index=time)
          resampled_labels = label.resample('6s').bfill()
          new_labels.append(resampled_labels[:len(agg1)].to_numpy())
    resampled = arr.resample('6s').bfill()
    new_labels = np.array(new_labels)
    np.save(destination_agg_resample_path + 'house_' + str(k) + '/aggregate_%d.npy' % i, resampled[:len(agg1)])
    np.save(destination_labels_resample_path + 'house_' + str(k) + '/strong_labels_%d.npy' % i,new_labels)
    