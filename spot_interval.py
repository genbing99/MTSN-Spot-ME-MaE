
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

# Check micro and macro peak score
def checkPeak(exp1_detected, exp2_detected, exp1_score, exp2_score, dif_threshold, k_exp):
    exp1_final = []
    for exp1_phase in exp1_detected:
        if exp1_phase[1] < len(exp2_score):
            if exp1_score[exp1_phase[1]] - exp2_score[exp1_phase[1]] > dif_threshold:
                exp1_final.append([exp1_phase[0], exp1_phase[1], exp1_phase[2]]) 
        else:
            if exp1_score[exp1_phase[1]] - exp2_score[-1] > dif_threshold:
                exp1_final.append([exp1_phase[0], exp1_phase[1], exp1_phase[2]]) 
    return exp1_final

def detectInterval(score_plot_agg, peak, left_dis, right_dis, threshold): # dis = distance to left and right of the peak
    start = 0
    best_diff = 0
    for left_index in range(peak-left_dis,peak+1):
        if left_index >= 0:
            diff = abs(score_plot_agg[peak] - score_plot_agg[left_index])
            if diff > best_diff and score_plot_agg[left_index] > threshold:
                start = left_index
                best_diff = diff
    end = min(peak + right_dis, len(score_plot_agg) - 1)
    best_diff = 0
    for right_index in range(peak,peak+right_dis+1):
        if right_index < len(score_plot_agg):
            diff = abs(score_plot_agg[peak] - score_plot_agg[right_index])
            if diff > best_diff and score_plot_agg[right_index] > threshold:
                end = right_index
                best_diff = diff
    return start, peak, end
    
# For score aggregation, to smooth the spotting confidence score
def smooth(y, box_pts):
    y = [each_y for each_y in y]
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth
    
def spotting(dataset_name, final_subjects, final_videos, dataset, result, subject_count, micro_threshold, macro_threshold, dif_threshold, show=False):
    color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if dataset_name == 'CAS_Test_cropped':
        micro_min = 5
        micro_max = 30
        macro_min = 15
        macro_max = 90
        micro_left_dis = 15
        micro_right_dis = 15
        macro_left_dis = 45
        macro_right_dis = 45
        k_micro = 6
        k_macro = 18
        peak_micro_dis = 12
        peak_macro_dis = 36
    elif dataset_name == 'SAMM_Test_cropped':
        micro_min = 60
        micro_max = 150
        macro_min = 100
        macro_max = 600
        micro_left_dis = 50
        micro_right_dis = 100
        macro_left_dis = 120
        macro_right_dis = 200
        k_micro = 37
        k_macro = 111
        peak_micro_dis = 600
        peak_macro_dis = 1200
        
    prev = 0
    videos = [ele for ele in final_videos for ele in ele]
    for videoIndex, video in enumerate(final_videos[subject_count-1]):
        preds_micro = []
        preds_macro = []
        micro_detected = []
        macro_detected = []
        countVideo = len([video for subject in final_videos[:subject_count-1] for video in subject])
        score_plot = np.array(result[prev:prev+len(dataset[countVideo+videoIndex])]) #Get related frames to each video
        
        # Micro Score aggregation
        score_plot_micro = score_plot.copy()
        for x in range(k_micro, len(score_plot_micro)-k_micro):
            score_plot_micro[x] = score_plot[x-k_micro:x+k_micro].mean()
        score_plot_micro = score_plot_micro[k_micro:-k_micro]
        
        # Macro Score aggregation
        score_plot_macro = score_plot.copy()
        for x in range(k_macro, len(score_plot_macro)-k_macro):
            score_plot_macro[x] = score_plot[x-k_macro:x+k_macro].mean()
        score_plot_macro = score_plot_macro[k_macro:-k_macro]

        #Plot the result to see the peaks
        #Note for some video the ground truth samples is below frame index 0 due to the effect of aggregation, but no impact to the evaluation
        if show:
            print('\nSubject:', final_subjects[subject_count-1], subject_count, 'Video:', videos[countVideo+videoIndex], countVideo+videoIndex)
            plt.figure(figsize=(15,3))
            plt.plot(score_plot_micro, color=color_list[0]) 
            plt.plot(score_plot_macro, color=color_list[3]) 
            plt.xlabel('Frame')
            plt.ylabel('Score')

        # Detect Micro
        peaks_micro, _ = find_peaks(score_plot_micro, height=micro_threshold, distance=peak_micro_dis)
        for peak in peaks_micro: 
            start, peak, end = detectInterval(score_plot_micro, peak, micro_left_dis, micro_right_dis, micro_threshold)
            if end-start > micro_min and end-start < micro_max and ( score_plot_micro[peak] > 0.95 or (score_plot_micro[peak] > score_plot_micro[start] and score_plot_micro[peak] > score_plot_micro[end])):
                micro_detected.append([start, peak, end])

        # Detect Macro
        peaks_macro, _ = find_peaks(score_plot_macro, height=macro_threshold, distance=peak_macro_dis)
        for peak in peaks_macro:
            start, peak, end = detectInterval(score_plot_macro, peak, macro_left_dis, macro_right_dis, macro_threshold)
            if end-start > macro_min and end-start < macro_max and ( score_plot_macro[peak] > 0.95 or (score_plot_macro[peak] > score_plot_macro[start] and score_plot_macro[peak] > score_plot_macro[end])):
                macro_detected.append([start, peak, end])
                    
        micro_detected = checkPeak(micro_detected, macro_detected, score_plot_micro, score_plot_macro, dif_threshold, k_micro)
        micro_list = []
        macro_list = []
                
        for micro_phase in micro_detected:
            preds_micro.append([micro_phase[0]+k_micro+1, 0, micro_phase[2]+k_micro+1, 0, 0, 0, 0])
            micro_list.append([micro_phase[0]+k_micro+1, micro_phase[2]+k_micro+1])
            plt.axvline(x=micro_phase[0], color=color_list[0])
            plt.axvline(x=micro_phase[2], color=color_list[0])
        for macro_phase in macro_detected:
            preds_macro.append([macro_phase[0]+k_macro+1, 0, macro_phase[2]+k_macro+1, 0, 0, 0, 0])
            macro_list.append([macro_phase[0]+k_macro+1, macro_phase[2]+k_macro+1])
            plt.axvline(x=macro_phase[0], color=color_list[3])
            plt.axvline(x=macro_phase[2], color=color_list[3])
    
            
        print('Micro Detected:')
        print(micro_list)
        print('Macro Detected:')
        print(macro_list)
        if show:
            print('Micro Before:', len(peaks_micro), 'After:', len(preds_micro))
            print('Macro Before:', len(peaks_macro), 'After:', len(preds_macro))
            plt.axhline(y=micro_threshold, color=color_list[0])
            plt.axhline(y=macro_threshold, color=color_list[3])
            plt.show()
            
        prev += len(dataset[countVideo+videoIndex])   
    return micro_list, macro_list