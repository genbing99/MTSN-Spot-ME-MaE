import time
import csv
import torch
import numpy as np
from torch.utils.data import DataLoader

from network import *
from dataloader import *
from spot_interval import *

def test(dataset_name, final_subjects, final_videos, dataset, X_all, X1_all, batch_size, micro_threshold, macro_threshold, dif_threshold):    
    # Write final result to csv file
    if dataset_name == 'CAS_Test_cropped':
        file_out = 'cas_pred.csv'
    elif dataset_name == 'SAMM_Test_cropped':
        file_out = 'samm_pred.csv'

    f = open(file_out, mode='w', newline='')
    writer = csv.writer(f, delimiter=',')

    # write header to the csv file
    writer.writerow(['vid', 'pred_onset', 'pred_offset', 'type'])
        
    start = time.time()
    batch_size = batch_size
    subject_count = 0
    result_final = []
    micro_total_detected = 0
    macro_total_detected = 0

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    model = MTSN().cuda()
    if dataset_name == 'CAS_Test_cropped':
        model.load_state_dict(torch.load("megc2022-pretrained-weights/cas_weights.pkl"))
    elif dataset_name == 'SAMM_Test_cropped':
        model.load_state_dict(torch.load("megc2022-pretrained-weights/samm_weights.pkl"))
    
    for subject_index in range(len(X_all)):
        print('Subject:', final_subjects[subject_count])
        subject_count += 1

        # Initialize training dataloader
        X = torch.Tensor(np.array(X_all[subject_index])).permute(0,3,1,2)
        X1 = torch.Tensor(np.array(X1_all[subject_index])).permute(0,3,1,2)
        test_dl = DataLoader(
            OFFSpottingDataset((X, X1)),
            batch_size=batch_size,
            shuffle=False,
        )

        # Testing
        model.eval()
        result_all = np.array([])
        for batch in test_dl:
            x1   = batch[0].to(device)
            x2   = batch[1].to(device)
            x3   = batch[2].to(device)
            x4   = batch[3].to(device)
            x5   = batch[4].to(device)
            x6   = batch[5].to(device)
            yhat = model(x1,x2,x3,x4,x5,x6).view(-1)
            result = yhat.cpu().data.numpy()
            result_all = np.append(result_all, result)

        # Spot intervals
        micro_list, macro_list = spotting(dataset_name, final_subjects, final_videos, dataset, result_all, subject_count, micro_threshold, macro_threshold, dif_threshold, show=False)
        result_final.append(result_all)
        
        for micro in micro_list:
            writer.writerow([str(final_subjects[subject_count-1]), micro[0], micro[1], 'me'])
            micro_total_detected += 1
        for macro in macro_list:
            writer.writerow([str(final_subjects[subject_count-1]), macro[0], macro[1], 'mae'])
            macro_total_detected += 1
        
    end = time.time()
    print('Total time taken for testing: ' + str(end-start) + 's')
    f.close()
    return result_final, micro_total_detected, macro_total_detected