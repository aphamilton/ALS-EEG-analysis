# -*- coding: utf-8 -*-

#Program to analyze the Tubingen data. The data has 10 subjects with ALS and 10 healthy controls.
#The program does three main things: 1) loads and transforms the EEG data in sensor space and source space, 
#                                    2) uses MNE functions to visualize it in various ways.
#                                    3) finds the LZC, MF, and ACW-50 for each subject.
#The functions for visualization automatically save a .png file for each subject.
#Below the functions there is a single file path for all of them to set.
#

#Import libraries
#region

#Libraries used for sensor space
import os
import numpy as np
import scipy
import mne
import os.path as op
import scipy.signal as signal
from statsmodels.tsa import stattools

#Libraries for source space
import matplotlib
from matplotlib import pyplot as plt
from mne.datasets import fetch_fsaverage
import vtk
import mayavi
import sobol_seq
import surfer
import nibabel
import PyQt5
import pyface
from mne.minimum_norm import make_inverse_operator, apply_inverse
import joblib
import threadpoolctl
import sklearn
import nilearn

print ("Finished loading libraries.")
#endregion

#Declare functions
#region

#Functions for raw; load, plot, print info
#region
def my_load_data():
    file_name = path_name + file_names_1[i] + str(j + 1) + file_names_2[i]
    raw = mne.io.read_raw_eeglab(file_name).crop(tmax = 291.92)    #Cropped at the length of the shortest file, the task files are much longer than this.
    print ("Loaded data for one subject.")
    return raw

def my_plot_power_spectra_channels_averaged(data_raw):
    fig = data_raw.plot_psd(1, 40, xscale='linear', average=True, show=False)
    fig.savefig(output_path + "Power_spectrum_averaged_" + str(i) + "_" + str(j) + ".png")
    print ("Finished a round of plotting power spectral density with all channels averaged.")

def my_plot_power_spectra_channels_separate(data_raw):
    fig = data_raw.plot_psd(1, 40, xscale='linear', average=False, show=False)
    fig.savefig(output_path + "Power_spectrum_channels_" + str(i) + "_" + str(j) + ".png")
    print ("Finished a round of plotting power spectral density with separate channels.")

#Plots sensor locations -- same for all subjects
def my_plot_sensor_locations(data_raw):
    fig = data_raw.plot_sensors(ch_type='eeg', show=False)
    fig.savefig(output_path + "Sensor_locations_" + str(i) + "_" + str(j) + ".png")
    print ("Finished plotting sensor locations for one subject.")

#Prints a topoplot showing a tiny power spectrum for each electrode (a terrible visualization in my opinion)
def my_topoplots_with_tiny_power_spectra(data_raw):
    fig = data_raw.plot_psd_topo(fmin=1, fmax=40, show=False)
    fig.savefig(output_path + "Topoplot_with_power_spectra_" + str(i) + "_" + str(j) + ".png")
    print ("Finished a topoplot.")

#Prints the raw EEG recording for each channel in an interactive plot
def my_plot_channels_separately(data_raw):
    current_title = "Temporary title"
    fig = data_raw.plot(title = current_title, n_channels = 20, show=False)
    fig.savefig(output_path + "Plot_of_channels_" + str(i) + "_" + str(j) + ".png")
    print ("Finished plotting raw EEG results for one subject.")

def my_all_raw_plots(raw):
    my_plot_power_spectra_channels_averaged(raw)
    my_plot_power_spectra_channels_separate(raw)
    #my_plot_sensor_locations(raw)
    #my_topoplots_with_tiny_power_spectra(raw)
    my_plot_channels_separately(raw)

#Print various info on the raw data from a single subject.
def my_raw_print_info(raw):
    print ("Printing info.")
    print (raw.info)
    print ("Printing annotations.")
    for k in raw._annotations:
        print (k)
    print ("Printing all member variables.")
    print (raw.__dict__)
#endregion

#Functions for events, epochs, and evoked; convert, plot
#region
def my_raw_to_events(raw):
    events,event_id = mne.events_from_annotations(raw, event_id=annotations_to_keep)
    print ("Found events for one subject.")
    return events

#In order to extract the stimulus/task events only, set annotation_number to 1 and cycle_length to 3.
#Only apply this function to the task data.
def my_create_events_stimulus_only(events, annotation_number, cycle_length):
    events_stimulus_only = []
    j = 0
    for event in events:
        if (event[2] == annotation_number):
            j += 1
            if (j == cycle_length):
                events_stimulus_only.append(event)
                j = 0
    return events_stimulus_only

def my_all_events_plots(events):
    print (events)
    fig = mne.viz.plot_events(events, show=False)
    fig.savefig(output_path + "Timing_of_events_" + str(i) + "_" + str(j) + ".png")

def my_events_to_epochs(raw, events):
    epochs = mne.Epochs(raw, events, event_repeated = "drop")
    print ("Created epochs for one subject.")
    return epochs

def my_visualize_epochs(epochs):
    fig = mne.viz.plot_epochs_image(epochs, combine="mean", show=False)
    fig[0].savefig(output_path + "Epochs_visualized_" + str(i) + "_" + str(j) + ".png")
    print ("Visualized epochs for one subject.")

def my_create_topoplots_by_frequency_band(epochs):
    fig = epochs.plot_psd_topomap(ch_type='eeg', normalize=True, show=False, vlim=(0,0.5))
    fig.savefig(output_path + "Topoplots_by_frequency_band_" + str(i) + "_" + str(j) + ".png")
    print ("Created a set of topoplots for one subject.")

def my_all_epochs_plots(epochs):
    print (epochs)
    my_visualize_epochs(epochs)
    my_create_topoplots_by_frequency_band(epochs)
    print ("Finished all epochs plots.")

def my_plot_evoked(data_evoked, type):
    fig = data_evoked.plot(gfp=True, spatial_colors=True, show=False)
    fig.savefig(output_path + "Evoked_plot_basic_" + type + "_" + str(i) + "_" + str(j) + ".png")

def my_plot_evoked_image(data_evoked, type):
    fig = data_evoked.plot_image(show=False)
    fig.savefig(output_path + "Evoked_plot_image_" + type + "_" + str(i) + "_" + str(j) + ".png")

def my_plot_compare_evokeds(data_evoked, type):
    fig = mne.viz.plot_compare_evokeds(data_evoked, show=False)
    fig[0].savefig(output_path + "Comparing_evokeds_" + type + "_" + str(i) + "_" + str(j) + ".png")

def my_all_evoked_plots(evoked, type):
    my_plot_evoked(evoked, type)
    my_plot_evoked_image(evoked, type)
    my_plot_compare_evokeds(evoked, type)
    print ("Created plots for the evoked dataset (may be one or more subjects).")
#endregion

#Functions for source space; convert, plot
#region
def my_create_volume_source_space():
    vol_src =   mne.setup_volume_source_space(subject, mri=mri, pos=10.0, bem=bem,
                subjects_dir=subjects_dir, add_interpolator=True, verbose=True)
    print ("Created volume source space.")
    return vol_src

#Creates a forward solution for one subject.
#The last paramater determines whether to create a surface-based forward solution or a volumetric inverse solution.
def my_create_forward_solution(evoked, which_src):
    fwd = mne.make_forward_solution(evoked.info, trans=trans, src=which_src, bem=bem, eeg=True, mindist=5.0, n_jobs=1)
    print (fwd)
    print ("Created forward solution for one subject.")
    return fwd

#Visualizes a forward solution (created by the previous function) for one subject.
def my_visualize_source_space_estimate(fwd):
    eeg_map = mne.sensitivity_map(fwd, ch_type='eeg', mode='fixed')
    brain = eeg_map.plot(surface='inflated', time_label='EEG sensitivity',
                         subjects_dir=subjects_dir, clim=dict(lims=[5, 50, 100]))
    brain.save_image(output_path + "Forward_solution_brain_" + str(i) + "_" + str(j) + ".png")
    print ("Created and saved an image of the the forward solution for one subject.")
    return brain

#This function's code is close to sample code available on the MNE website. It creates an inverse solution.
#The last parameter is a Bool that determines whether to create a surface-based inverse solution or a volumetric inverse solution.
def my_create_inverse_solution(epochs, evoked, fwd, is_volume):
    noise_cov = mne.compute_covariance(epochs, tmax=0., method=['empirical'], rank=None, verbose=True)
    if is_volume == True:
        loose = 1.0
    else:
        loose = 0.2
    inverse_operator = make_inverse_operator(evoked.info, fwd, noise_cov, loose=loose, depth=0.8)
    method = "dSPM"
    snr = 3.
    lambda2 = 1. / snr ** 2
    stc = apply_inverse(evoked, inverse_operator, lambda2, method=method,
                    pick_ori=None, return_residual=False, verbose=True)
    print ("Created an inverse solution for one subject.")
    return stc

def my_plot_surface_source_estimates(stc):
    initial_time = 0
    brain =     stc.plot(subjects_dir=subjects_dir, initial_time=initial_time,
                clim=dict(kind='value', lims=[0, 2, 4]))
    brain.save_image(output_path + "Surface_source_estimate_" + str(i) + "_" + str(j) + ".png")
    print ("Created a plot of surface source estimates for one subject.")

def my_plot_volume_source_estimates(stc, vol_src):
    print (stc)
    brain = stc.plot(vol_src, subject='fsaverage', subjects_dir=subjects_dir, show=False)
    brain.savefig(output_path + "Volume_source_estimate_" + str(i) + "_" + str(j) + ".png")
    print ("Created a plot of volume source estimates for one subject.")
#endregion

#Complex dynamic measures (MF, ACW, PLE, LZC)
#region

# MF script adapted from Mehrshad
#region
def my_calc_mf(freq, psd):
    cumulative_power_sum = np.cumsum(psd, axis=1)
    total_power = cumulative_power_sum[:, -1]
    half_total_power = total_power / 2
    half_total_power_2D = half_total_power.reshape(-1, 1)     #Convert 1D array to 2D array with 1 column
    is_power_more_than_half = cumulative_power_sum >= half_total_power_2D
    first_index_in_second_half = np.argmax(is_power_more_than_half, axis=1)
    #return freq[first_index_in_second_half]
    
    #All the values generated by the MF function above were from a few discrete values.
    #I created this extension to interpolate values in a way that I believe is logical,
    #but not being an expert on MF I encourage you to take it with a grain of salt.
    #If you comment it out and comment in the last line above, the function works.
    row_indexes = np.arange(len(cumulative_power_sum))
    cumulative_power_just_above_median = cumulative_power_sum[row_indexes, first_index_in_second_half]
    cumulative_power_just_below_median = cumulative_power_sum[row_indexes, first_index_in_second_half - 1]
    interpolation_factor = (half_total_power - cumulative_power_just_below_median) / (cumulative_power_just_above_median - cumulative_power_just_below_median)
    frequency_just_above_median = freq[first_index_in_second_half]
    frequency_just_below_median = freq[first_index_in_second_half - 1]
    interpolated_frequency = (frequency_just_below_median * (1 - interpolation_factor)) + (frequency_just_above_median * interpolation_factor)
    return interpolated_frequency

# ACW-50 script adapted from Mehrshad -- working
#region

def calc_a_acf(ts, n_lag=None, fast=True):
     if not n_lag:
         n_lag = len(ts)
     return stattools.acf(ts, nlags=n_lag, qstat=False, alpha=None, fft=fast)
 
def calc_a_acw(ts, n_lag=None, fast=True, is_acf=False):
     acf = ts if is_acf else calc_a_acf(ts, n_lag, fast)
     return 2 * np.argmax(acf < 0.5) - 1

def my_calculate_acw_50(raw):
    raw_times = np.array([0, 291.92])
    start_sample, stop_sample = (raw_times*raw.info['sfreq']).astype(int)
    data = raw[0 : raw.info['nchan'], start_sample : stop_sample][0]
    acw_50 = [None]*data.shape[0]
    i = 0
    for ts in data:
        acw_50[i] = calc_a_acw(ts)
        i += 1
    return acw_50

#endregion

# LZC script adapated from Mehrshad -- working
#region

def lempel_ziv_complexity(binary_sequence):
     u, v, w = 0, 1, 1
     v_max = 1
     length = len(binary_sequence)
     complexity = 1
     while True:
         if binary_sequence[u + v - 1] == binary_sequence[w + v - 1]:
             v += 1
             if w + v >= length:
                 complexity += 1
                 break
         else:
             if v > v_max:
                 v_max = v
             u += 1
             if u == w:
                 complexity += 1
                 w += v_max
                 if w > length:
                     break
                 else:
                     u = 0
                     v = 1
                     v_max = 1
             else:
                 v = 1
     return complexity
 
def my_calculate_lzc(raw, t_start, t_end):    #Maximum range is 0 to 291.92.
    raw_times = np.array([t_start, t_end])
    start_sample, stop_sample = (raw_times*raw.info['sfreq']).astype(int)
    n_sample = stop_sample - start_sample
    print (n_sample)
    norm_factor = n_sample / np.log2(n_sample)
    data = raw[0 : raw.info['nchan'], start_sample : stop_sample][0]
    lzc = [None]*data.shape[0]
    # ts is a time series
    i = 0
    for ts in data:
        print ("Calculating LZC for electrode number " + str(i) + ".")
        bin_ts = np.char.mod('%i', ts >= np.median(ts))
        lzc[i] = lempel_ziv_complexity("".join(bin_ts)) / norm_factor
        i += 1
    return lzc

#endregion

# PLE adapted from David (no filter version) -- not working for me
#region

def ple(data):
     print (data)
#     f, pxx = signal.welch(data, fs=200, window='hanning', nperseg=400,
#                           noverlap=200, scaling='density', average='mean')
     psds, freqs = mne.time_frequency.psd_welch(data)
     log_freqs = np.log(freqs)
     log_psds = np.log(psds)
     index = np.isfinite(log_freqs) & np.isfinite(log_psds)
     print ("Printing index: " + str(index))
     polynomial_coefficients = np.polyfit(log_freqs[index], log_psds[index], 1)
     print(polynomial_coefficients)

#endregion

#endregion
#endregion
#endregion

#Declare variables and arrays
#region
NUMBER_OF_DATASETS_AVAILABLE = 5
DATASET_SIZE = [10, 10, 9, 10, 9]       #Maximum is [10, 10, 9, 10, 9]
raw = [None]*NUMBER_OF_DATASETS_AVAILABLE
for i in range(NUMBER_OF_DATASETS_AVAILABLE):
    raw[i] = [None]*DATASET_SIZE[i]
SHOULD_LOAD = [False, True, True, True, True]
NUMBER_OF_DATASETS_LOADED = 0
TOTAL_NUMBER_OF_SUBJECTS = 0
for i in range (NUMBER_OF_DATASETS_AVAILABLE):
    if SHOULD_LOAD[i] == True:
        NUMBER_OF_DATASETS_LOADED += 1
        TOTAL_NUMBER_OF_SUBJECTS += DATASET_SIZE[i]
print ("Will load " + str(NUMBER_OF_DATASETS_LOADED) + " datasets.")
print (str(TOTAL_NUMBER_OF_SUBJECTS) + " subjects will be loaded in total.")

path_name = "C:/Users/Arthur/Documents/EEG Analysis - Python/EEG Data Files/"
file_names_1 = ["Tubingen ALS 0.1 Hz Rest/SELFA",   #I haven't used this one for anything.
                "Tubingen ALS 1 Hz Rest/SELFA",
                "Tubingen ALS 1 Hz Task/SELFA",
                "Tubingen HC 1 Hz Rest/SELFC",
                "Tubingen HC 1 Hz Task/SelfC"]
file_names_2 = ["S001R01.dat_rest_150Hz.set_preprocessed.set",
                "S001R01.dat_rest.set",
                "S001R03.dat_task.set",
                "S001R01.dat_rest.set",
                "S001R03.dat_rest.set"]     #The files for this group are labelled "rest"
                                            #for some reason but they are really task data.

output_path = "Output/Newest/"
full_output_path = "C:/Users/Arthur/Documents/EEG Analysis - Python/Output/Newest/"

annotations_to_keep = {'PresentationPhase':1,    #PresentationPhase is the only one you will need. It shows the three phases of each trial for the task data.
                       'TrialC':2,
                       'CurrentTrial':3,
                       'CurrentBlock':4,
                       'ExpState':5
                       }

all_evoked = [None]*NUMBER_OF_DATASETS_AVAILABLE
for i in range(NUMBER_OF_DATASETS_AVAILABLE):
    all_evoked[i] = [None]*DATASET_SIZE[i]
all_grand_averaged = [None]*TOTAL_NUMBER_OF_SUBJECTS
next_index = 0

all_mf = [None]*NUMBER_OF_DATASETS_AVAILABLE
for i in range(NUMBER_OF_DATASETS_AVAILABLE):
    all_mf[i] = [None]*DATASET_SIZE[i]

all_acw_50 = [None]*NUMBER_OF_DATASETS_AVAILABLE
for i in range(NUMBER_OF_DATASETS_AVAILABLE):
    all_acw_50[i] = [None]*DATASET_SIZE[i]

all_lzc = [None]*NUMBER_OF_DATASETS_AVAILABLE
for i in range(NUMBER_OF_DATASETS_AVAILABLE):
    all_lzc[i] = [None]*DATASET_SIZE[i]

#Preliminary code to load files for entering source space. These are all MNE defaults b/c there is no MRI data for the dataset.
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)
subject = 'fsaverage'
trans = 'fsaverage' 
src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
mri = op.join(fs_dir, 'mri', 'T1.mgz')
vol_src = my_create_volume_source_space()
#endregion

#Main body of program
#region

for i in range(NUMBER_OF_DATASETS_AVAILABLE):
    if SHOULD_LOAD[i] == True:
        all_evoked = [None]*DATASET_SIZE[i]
        for j in range(DATASET_SIZE[i]):
            raw = my_load_data()
            #print (raw.info)
            raw.set_eeg_reference('average', projection=True) #I believe this is only needed for source space.

            psds, freqs = mne.time_frequency.psd_welch(raw)
            print ("Calculated PSD for one subject.")
            med_freqs = my_calc_mf(freqs, psds)
            mf_across_channels = np.mean(med_freqs)
            print ("Calculated arithmetic mean of MF across channels for one subject.")
            print (mf_across_channels)            

            acw_50 = my_calculate_acw_50(raw)
            print ("Calculated ACW for one subject.")
            print (acw_50)
            acw_50_across_channels = np.mean(acw_50)
            print ("Calculated arithmetic mean of ACW-50 across channels for one subject.")
            print (acw_50_across_channels)

            lzc = my_calculate_lzc(raw, 0, 10)
            print ("Calculated LZC for one subject.")
            print (lzc)
            lzc_across_channels = np.mean(lzc)
            print ("Calculated arithmetic mean of LZC across channels for one subject.")
            print (lzc_across_channels)

            all_mf[i][j] = mf_across_channels
            all_acw_50[i][j] = acw_50_across_channels
            all_lzc[i][j] = lzc_across_channels

            my_all_raw_plots(raw)
            events = my_raw_to_events(raw)
            if (i == 2 or i == 4):     #This is only for the task data. It extracts only the events for the stimulus/task onset.
                events = my_create_events_stimulus_only(events, 1, 3)
            epochs = my_events_to_epochs(raw, events)
            my_all_events_plots(events)
            evoked = epochs.average()
            my_all_epochs_plots(epochs)
            my_all_evoked_plots(evoked, "individual")

            #Create and visualize surface source space
            surface_fwd = my_create_forward_solution(evoked, src)
            my_visualize_source_space_estimate(surface_fwd)
            stc_surface = my_create_inverse_solution(epochs,evoked,surface_fwd,False)
            my_plot_surface_source_estimates(stc_surface)

            #Create and visualize volume source space
            volume_fwd = my_create_forward_solution(evoked, vol_src)
            stc_volume = my_create_inverse_solution(epochs,evoked,volume_fwd,True)
            my_plot_volume_source_estimates(stc_volume, vol_src)
            
            #Convert volumetric inverse solution to .nii file.
            #Note that some of Mehrshad's functions need a .nii file, but it is a different type with the same file extension.
            nifti = stc_volume.as_volume(vol_src, format="nifti2")
            print ("Created an object of type nifti for one subject.")
            nibabel.save(nifti, output_path + "Nifti_" + str(i) + "_" + str(j) + ".nii")
            print ("Saved a nifti file.")

            all_evoked[j] = evoked
            all_grand_averaged[next_index] = evoked
            next_index += 1
        #Create a grand average of all files from one dataset, and plot it.
        #The file naming for the grand average plots doesn't work quite right.
        grand_averaged = mne.grand_average(all_evoked)
        print ("Created grand average for one group of subjects.")
        my_all_evoked_plots(grand_averaged, "grand_average")
#Create a grand average of all the files from all the datasets you loaded, and plot it.
#The file naming for the grand average plots doesn't work quite right.
grand_average_across_datasets = mne.grand_average(all_grand_averaged)
print ("Created a single grand average for all the data that was loaded.")
my_all_evoked_plots(grand_average_across_datasets, "grand_grand_average")

print (all_mf)
print ("Printed MF values for all subjects.")
print (all_acw_50)
print ("Printed ACW-50 values for all subjects.")
print (all_lzc)
print ("Printed LZC values for all subjects.")

#endregion
