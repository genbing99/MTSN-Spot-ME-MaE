import argparse
from distutils.util import strtobool

from prepare_data import *
from train_model import *
from load_gt import *

def main(config):

    dataset_name = config.dataset_name
    print('\n ------', dataset_name, '------')

    # Load Training Data
    print('\n ------ Loading Processed Training Data ------')
    final_subjects, final_videos, dataset, dataset1 = load_train_data(dataset_name)

    # Load Ground Truth Label
    print('\n ------ Loading Excel ------')
    codeFinal = load_excel(dataset_name)
    print('\n ------ Loading Ground Truth From Excel ------')
    final_subjects, final_videos, final_samples, final_exp = load_label(final_subjects, final_videos, codeFinal) 

    # Spotting Pseudo-labeling
    print('\n ------ Pseudo-Labeling ------')
    X, X1, Y, Y1 = pseudo_labeling(dataset, dataset1, final_samples)

    # Setting paramaters
    batch_size = 128
    lr_classifier = 0.00005
    epochs = 10
    ratio = 1

    # Training & Evaluation
    print('\n ------ MTSN Training ------')
    train(dataset_name, X, X1, Y, Y1, epochs, lr_classifier, batch_size, ratio)

    print('\n ------ Completed ------')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--dataset_name', type=str, default='CASME_sq') # CASME_sq or SAMMLV only
    
    config = parser.parse_args()

    main(config)