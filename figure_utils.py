import json
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os
import re
import collections
plt.style.use('ggplot')

def find_infreq_index(j_data):
    infreq_indexes =[]
    names = j_data['net']['params']['connParams']

    infreq_stims = [s for s in names.keys() if "dev" in s]
    for stim in infreq_stims:
        infreq_indexes.append(int(stim[9]))

    return infreq_indexes

def open_file_as_json(name):
    with open(name) as f:
        data = f.read()

    return json.loads(data)

### ploting the frequnt vs infrequent LFPs
def prepare_data_LFP(LFP_dict, N_peaks, infreq_index, ms_to_trim=5, mean=True):

    # avg 0,1 electrodes
    avg_LFP = [np.mean(i) for i in LFP_dict]

    #create a matrix and populate with the LFP data by peak
    LFP_peak_matrix = np.zeros(shape=(N_peaks,10000))

    for i in range(N_peaks):
        LFP_peak_matrix[i]=avg_LFP[i*10000:(i+1)*10000]
        #remove initial peak
        LFP_peak_matrix[i][:500]=LFP_peak_matrix[i][501]
        trimmed= int(5000+ (ms_to_trim*100))
        LFP_peak_matrix[i][5000:trimmed]=LFP_peak_matrix[i][5000]

    # seperate freq vs infreq peaks
    peak_freq = np.delete(LFP_peak_matrix, infreq_index, 0)
    if mean:
        peak_freq_mean = np.mean(peak_freq, axis=0)
    if not mean:
        peak_freq_mean=peak_freq

    peak_infreq = LFP_peak_matrix[infreq_index]

    return {'infreq':peak_infreq, 'freq':peak_freq_mean}

def exctract_data_LFP(file_names_list, N_peaks):
    all_infreq_LFPs = []
    all_freq_LFP = []

    # import all files LFP data
    for file_name in file_names_list:

        j_data = open_file_as_json(file_name)

        LFP = j_data['simData']['LFP']

        freq_stim = j_data['net']['params']['connParams']['Stim_std_0->PYR4']['postConds']['x']

        names = j_data['net']['params']['connParams']

        infreq_stim = find_infreq_index(j_data)


        prepared_data = prepare_data_LFP(LFP, N_peaks, infreq_stim[0], 5, True)

        all_infreq_LFPs.append(prepared_data['infreq'])
        all_freq_LFP.append(prepared_data['freq'])

    mean_infreq_LFPs = np.mean(all_infreq_LFPs, axis=0)
    mean_freq_LFP = np.mean(all_freq_LFP, axis=0)

    return {'infreq':mean_infreq_LFPs, 'freq':mean_freq_LFP}

def plot_freq_vs_infreq_LFP (PATH_LIST, n_stim):
    data = exctract_data_LFP(PATH_LIST, n_stim)

    plt.plot(-1*data['freq'], label='frequent', c='grey')
    plt.plot(-1*data['infreq'], label='infrequent', c='coral', alpha=.7)
    plt.title('Frequent vs Infrequent mean potentials')
    plt.legend()
    plt.show()

def plot_full_LFP(file_names_list, N_peaks):
    N_msrmnt = 10000

    for file_name in file_names_list:
        j_data = open_file_as_json(file_name)
        infreq_stim = find_infreq_index(j_data)
        LFP = j_data['simData']['LFP']
        ms_to_trim=5
        prepared_data = prepare_data_LFP(LFP, N_peaks, infreq_stim[0], ms_to_trim ,False)

        all_infreq_LFPs=prepared_data['infreq']
        all_freq_LFP=prepared_data['freq']

        #avg_LFP = [np.mean(i) for i in LFP]
        all_peaks=[]
        for peak in range(N_peaks-1):
            if peak==infreq_stim[0]:
                all_peaks.append(all_infreq_LFPs)
                all_peaks.append(1000*[all_infreq_LFPs[-1]])
                all_peaks.append(all_freq_LFP[peak])
            else:
                all_peaks.append(all_freq_LFP[peak])
            all_peaks.append(1000*[all_freq_LFP[peak][-1]])

        flat_peaks= [-1*item for sublist in all_peaks for item in sublist]

        N_per_stim=len(all_infreq_LFPs)+1000

        #plt.plot(all_infreq_LFPs, c='black')
        plt.plot(flat_peaks, c='grey')
        plt.plot(range(infreq_stim[0]*N_per_stim,(infreq_stim[0]+1)*N_per_stim),
                            flat_peaks[infreq_stim[0]*N_per_stim:(infreq_stim[0]+1)*N_per_stim],
                            c='coral')
        plt.title('Layer 2/3 LFP  (trimmed first {} ms)'.format(ms_to_trim))
        plt.show()

def get_spk_data(file_name):
    j_data = open_file_as_json(file_name)

    spktime = j_data['simData']['spkt']
    spkid = j_data['simData']['spkid']
    infreq_indexes = find_infreq_index(j_data)

    return {'spktime':spktime, 'spkid':spkid, 'infreq_index':infreq_indexes}

def plot_num_of_neurons(path, trim_ms, N_peaks):
    data = get_spk_data(path)

    spktime = data['spktime']
    spkid = data['spkid']
    infreq_id = data['infreq_index'][0]
    # count spike ids per spike times
    all_spk_times={t:[] for t in list(set(spktime))}
    for spk_t, spk_id in zip(spktime, spkid):
        all_spk_times[spk_t].append(spk_id)

    #sort and remove first 5ms
    spks_per_t = collections.OrderedDict(sorted({t:
        np.sum(all_spk_times[t]) for t in all_spk_times}.items()))

    X={i:[] for i in range(N_peaks)}
    Y={i:[] for i in range(N_peaks)}
    for t, n_spikes in spks_per_t.items():
        stim_i = t//1000

        # delete imitial 5 ms
        if (((t>stim_i*1000+500) and (t<stim_i*1000+500+trim_ms)) or  (t<500)):
            pass
        else:
            X[stim_i].append(t)
            Y[stim_i].append(n_spikes)
    label_ploted=False
    for stim in range(N_peaks):


        if stim==infreq_id:
            c='coral'
            label='infrequent'
        else:
            c='grey'
            if label_ploted:
                label=None
            else:
                label='frequent'
                label_ploted=True


        plt.plot(X[stim][100:],Y[stim][100:], c=c, label=label)

    plt.title('Layer 2/3+4 # of recruited neurons')
    plt.legend()
    plt.show()
###################

path='/Users/gilikarni/Google Drive/work/TU-berlin/Capstone/Code/output_files/simple_model_pyr4_bask23_G_3_W_000015_pyr23_g_2_w_3_dev_loc.json'
# for more than one json , list the paths
#plot_freq_vs_infreq_LFP([path], 8)
#plot_full_LFP([path],8)
plot_num_of_neurons(path,50,8)

D= open_file_as_json(path)
print (D.keys())
print (D['simData'].keys())
print (D['net']['params'].keys())

print (D['net']['cells'][:2])
print (len(D['simData']['LFP'])) #80000 as the length of the sim in 10*ms
print (len(D['simData']['spkid'])) ### 15570
print (len(set(D['simData']['spkid']))) #810
print (len(D['simData']['spkt'])) ### 15570
print (len(set(D['simData']['spkt']))) #12153
