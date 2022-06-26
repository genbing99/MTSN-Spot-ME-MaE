import pickle

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