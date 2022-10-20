import pandas as pd
import numpy as np

def load_excel(dataset_name):
    if(dataset_name == 'CASME_sq'):
        xl = pd.ExcelFile('megc2021-ground-truth/CASME_sq_gt.xlsx') #Specify directory of excel file

        colsName = ['subject', 'video', 'onset', 'apex', 'offset', 'au', 'emotion', 'type', 'selfReport']
        codeFinal = xl.parse(xl.sheet_names[0], header=None, names=colsName) #Get data

        videoNames = []
        for videoName in codeFinal.iloc[:,1]:
            videoNames.append(videoName.split('_')[0])
        codeFinal['videoName'] = videoNames

        naming1 = xl.parse(xl.sheet_names[2], header=None, converters={0: str})
        dictVideoName = dict(zip(naming1.iloc[:,1], naming1.iloc[:,0]))
        codeFinal['videoCode'] = [dictVideoName[i] for i in codeFinal['videoName']]

        naming2 = xl.parse(xl.sheet_names[1], header=None)
        dictSubject = dict(zip(naming2.iloc[:,2], naming2.iloc[:,1]))
        codeFinal['subjectCode'] = [dictSubject[i] for i in codeFinal['subject']]
        
    elif(dataset_name=='SAMMLV'):
        xl = pd.ExcelFile('megc2021-ground-truth/SAMMLV_gt.xlsx')

        colsName = ['Subject', 'Filename', 'Inducement Code', 'Onset', 'Apex', 'Offset', 'Duration', 'Type', 'Action Units', 'Notes']
        codeFinal = xl.parse(xl.sheet_names[0], header=None, names=colsName, skiprows=[0,1,2,3,4,5,6,7,8,9])

        videoNames = []
        subjectName = []
        for videoName in codeFinal.iloc[:,1]:
            videoNames.append(str(videoName).split('_')[0] + '_' + str(videoName).split('_')[1])
            subjectName.append(str(videoName).split('_')[0])
        codeFinal['videoCode'] = videoNames
        codeFinal['subjectCode'] = subjectName
        codeFinal.rename(columns={'Type':'type', 'Onset':'onset', 'Offset':'offset', 'Apex':'apex'}, inplace=True) 
        
    return codeFinal

def load_label(dataset_name, subjects, subjectsVideos, codeFinal):
    if dataset_name == 'CASME_sq':
        dataset_expression_type = 'micro-expression'
    else:
        dataset_expression_type = 'Micro - 1/2'
        
    exp = []
    vid_need = []
    vid_count = 0
    micro = 0
    macro = 0
    ground_truth = []
    for sub_video_each_index, sub_vid_each in enumerate(subjectsVideos):
        ground_truth.append([])
        exp.append([])
        for videoIndex, videoCode in enumerate(sub_vid_each):
            on_off = []
            exp_subject = []
            for i, row in codeFinal.iterrows():
                if (row['subjectCode']==subjects[sub_video_each_index]): #S15, S16... for CAS(ME)^2, 001, 002... for SAMMLV
                    if (row['videoCode']==videoCode):
                        if (row['type']==dataset_expression_type): #Micro-expression or macro-expression
                            micro += 1
                            exp_subject.append('micro')
                        else:
                            macro += 1
                            exp_subject.append('macro')
                        if (row['offset']==0): #Take apex if offset is 0
                            on_off.append([int(row['onset']-1), int(row['apex']-1), int(row['apex']-1)])
                        else:
                            if(dataset_expression_type!='Macro' or int(row['onset'])!=0): #Ignore the samples that is extremely long in SAMMLV
                                on_off.append([int(row['onset']-1), int(row['apex']-1), int(row['offset']-1)])
            if(len(on_off)>0):
                vid_need.append(vid_count) #To get the video that is needed
            ground_truth[-1].append(on_off) 
            exp[-1].append(exp_subject)
            vid_count+=1

    #Remove unused video
    final_samples = []
    final_videos = []
    final_subjects = []
    final_exp = []
    count = 0
    for subjectIndex, subject in enumerate(ground_truth):
        final_samples.append([])
        final_videos.append([])
        final_exp.append([])
        for samplesIndex, samples in enumerate(subject):
            if (count in vid_need):
                final_samples[-1].append(ground_truth[subjectIndex][samplesIndex])
                final_videos[-1].append(subjectsVideos[subjectIndex][samplesIndex])
                final_subjects.append(subjects[subjectIndex])
                final_exp[-1].append(exp[subjectIndex][samplesIndex])
            count += 1

    #Remove the empty data in array
    final_subjects = np.unique(final_subjects)
    final_videos = [ele for ele in final_videos if ele != []]
    final_samples = [ele for ele in final_samples if ele != []]
    final_exp = [ele for ele in final_exp if ele != []]

    print('Final Ground Truth Data')
    print('Subjects Name', final_subjects)
    print('Videos Name: ', final_videos)
    print('Samples [Onset, Offset]: ', final_samples)
    print('Expression Type:', final_exp)
    print('Total Micro:', micro, 'Macro:', macro)

    return final_subjects, final_videos, final_samples, final_exp

