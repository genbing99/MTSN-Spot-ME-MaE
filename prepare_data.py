import pickle
import numpy as np

# Load data from pre-computed optical flow
def load_data(dataset_name):
    final_videos = pickle.load( open( "megc2022-processed-data/" + dataset_name + "_subjectsVideos.pkl", "rb" ) )
    final_subjects = pickle.load( open( "megc2022-processed-data/" + dataset_name + "_subjects.pkl", "rb" ) )
    if dataset_name == 'CAS_Test_cropped':
        dataset = pickle.load( open( "megc2022-processed-data/" + dataset_name + "_dataset_k6.pkl", "rb" ) )
    elif dataset_name == 'SAMM_Test_cropped':
        dataset = pickle.load( open( "megc2022-processed-data/" + dataset_name + "_dataset_k37.pkl", "rb" ) )
    dataset1 = pickle.load( open( "megc2022-processed-data/" + dataset_name + "_dataset_k1.pkl", "rb" ) )
    return final_subjects, final_videos, dataset, dataset1

def prepare_dataset(dataset, dataset1):
    X_all = []
    X1_all = []
    for video_index, video in enumerate(dataset):
        X_all.append(video)
    for video_index, video in enumerate(dataset1):
        X1_all.append(video[:len(dataset[video_index])])
    return X_all, X1_all

# Load training data from pre-computed optical flow
def load_train_data(dataset_name):
    final_videos = pickle.load( open( "megc2021-processed-data/" + dataset_name + "_subjectsVideos.pkl", "rb" ) )
    final_subjects = pickle.load( open( "megc2021-processed-data/" + dataset_name + "_subjects.pkl", "rb" ) )
    if dataset_name == 'CASME_sq':
        dataset = pickle.load( open( "megc2021-processed-data/" + dataset_name + "_dataset_k6.pkl", "rb" ) )
    elif dataset_name == 'SAMMLV':
        dataset = pickle.load( open( "megc2021-processed-data/" + dataset_name + "_dataset_k37.pkl", "rb" ) )
    dataset1 = pickle.load( open( "megc2021-processed-data/" + dataset_name + "_dataset_k1.pkl", "rb" ) )
    return final_subjects, final_videos, dataset, dataset1

    
def pseudo_labeling(dataset, dataset1, final_samples):
    pseudo_y = []
    pseudo_y1 = []
    video_count = 0 

    for subject_index, subject in enumerate(final_samples):
        for video_index, video in enumerate(subject):
            samples_arr = []
            pseudo_y_each = [0]*(len(dataset[video_count]))
            pseudo_y1_each = [0]*(len(dataset[video_count]))

            for sample_index, sample in enumerate(video):
                if sample[0] > 1: # SAMMLV has few samples with onset 0 or 1 (with super long duration)
                    onset = sample[0]
                else:
                    onset = sample[1]
                offset = sample[2]+1
                for frame_index, frame in enumerate(range(onset, offset)):
                    if frame < len(pseudo_y_each):
                        # Hard label
                        pseudo_y_each[frame] = 1
                        # Soft label
                        if frame >= sample[0] and frame <= sample[1]:
                            score = frame_index / (sample[1] - sample[0])
                            pseudo_y1_each[frame] = score
                        elif frame > sample[1] and frame <= sample[2]:
                            score = (sample[2] - frame) / (sample[2] - sample[1])
                            pseudo_y1_each[frame] = score
            pseudo_y.append(pseudo_y_each)
            pseudo_y1.append(pseudo_y1_each)
            video_count+=1

    print('Total video:', len(pseudo_y))

    # Integrate all videos into one dataset
    pseudo_y = [y for x in pseudo_y for y in x]
    pseudo_y1 = [y for x in pseudo_y1 for y in x]
    print('Total frames:', len(pseudo_y))

    Y = np.array(pseudo_y)
    Y1 = np.array(pseudo_y1)
    X = []
    X1 = []
    for video_index, video in enumerate(dataset):
        X.append(video)
    for video_index, video in enumerate(dataset1):
        X1.append(video[:len(dataset[video_index])])
    X = [frame for video in X for frame in video]
    X1 = [frame for video in X1 for frame in video]
    print('\nTotal X:', len(X), 'Total X1:', len(X1), ', Total y:', len(Y), ', Total y:', len(Y1))
    
    return X, X1, Y, Y1
    