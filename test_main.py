from prepare_data import *
from test_model import *

batch_size = 128

# Setting paramaters for CAS_Test
dataset_name = 'CAS_Test_cropped' 
dif_threshold = 0.4
micro_threshold = 0.7
macro_threshold = 0.7
print('Start dataset CAS_Test')

# Load from preprocessed data
final_subjects, final_videos, dataset, dataset1 = load_data(dataset_name)
print('Done loading data')

# Get data
X_all, X1_all = prepare_dataset(dataset, dataset1)
print('Done preparing data')

# Get spotted intervals for the dataset
result_final, micro_total_detected, macro_total_detected = test(dataset_name, final_subjects, final_videos, dataset, X_all, X1_all, batch_size, micro_threshold, macro_threshold, dif_threshold)
print('Micro detected:', micro_total_detected, '| Macro detected:', macro_total_detected)
print('Done spotting CAS_Test')

######################################################################################

# Setting paramaters for SAMM_Test
dataset_name = 'SAMM_Test_cropped' 
dif_threshold = 0.37
micro_threshold = 0.5
macro_threshold = 0.5
print('Start dataset SAMM_Test')

# Load from preprocessed data
final_subjects, final_videos, dataset, dataset1 = load_data(dataset_name)
print('Done loading data')

# Get data
X_all, X1_all = prepare_dataset(dataset, dataset1)
print('Done preparing data')

# Get spotted intervals for the dataset
result_final, micro_total_detected, macro_total_detected = test(dataset_name, final_subjects, final_videos, dataset, X_all, X1_all, batch_size, micro_threshold, macro_threshold, dif_threshold)
print('Micro detected:', micro_total_detected, '| Macro detected:', macro_total_detected)
print('Done spotting SAMM_Test')