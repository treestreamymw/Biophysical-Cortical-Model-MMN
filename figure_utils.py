import json
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os
import re
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
def prepare_data_LFP(LFP_dict, N_peaks, infreq_index):

    # avg 0,1 electrodes
    avg_LFP = [np.mean(i) for i in LFP_dict]

    #create a matrix and populate with the LFP data by peak
    LFP_peak_matrix = np.zeros(shape=(N_peaks,10000))
    for i in range(N_peaks):
        LFP_peak_matrix[i]=avg_LFP[i*10000:(i+1)*10000]

    # seperate freq vs infreq peaks
    peak_freq = np.delete(LFP_peak_matrix, infreq_index, 0)
    peak_freq_mean = np.mean(peak_freq, axis=0)

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


        for stim in range(N_peaks):
            if stim not in infreq_stim:
                current_stim =  j_data['net']['params']['connParams']['Stim_{}_{}->PYR4'.format('std', stim)]['postConds']['x']
            else:
                current_stim =  j_data['net']['params']['connParams']['Stim_{}_{}->PYR4'.format('dev', stim)]['postConds']['x']

            if current_stim != freq_stim:
                infreq_index = stim
                break

        prepared_data = prepare_data_LFP(LFP, N_peaks, infreq_index)

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


## plotting the individual spikes



def get_spk_data(file_name):
    j_data = open_file_as_json(file_name)

    spktime = j_data['simData']['spkt']
    spkid = j_data['simData']['spkid']
    infreq_indexes = find_infreq_index(j_data)

    return {'spktime':spktime, 'spkid':spkid, 'infreq_index':infreq_indexes}

def generate_data_spks(spktime, n_spike):
    n = (n_spike+1)*100

    round_spktime = {round(i/n,2):0 for i in spktime}
    for j in spktime:
        round_spktime[round(j/n,2)]+=1

    X = [i/100 for i in range(1,n+1)]
    Y = [round_spktime.get(i) or 0 for i in X]

    #X1= np.mean(np.array(X).reshape(-1, 5), axis=1)
    #Y1= np.mean(np.array(Y).reshape(-1, 5), axis=1)

    return [X,Y]

def create_plot_spks (x, y, infreq_id):
    N = 100

    infreq_id = infreq_id[0]+1
    alpha = .6
    width=.02
    plt.bar(x[:(infreq_id)*N],y[:(infreq_id)*N], alpha=alpha, color='black',
            label='frequent', width=width, align ='edge')
    plt.bar(x[(infreq_id)*N:(infreq_id+1)*N],y[(infreq_id)*N:(infreq_id+1)*N],\
             label='infrequent', color='#F8766D', width=width, align ='edge')
    plt.bar(x[(infreq_id+1)*N:],y[(infreq_id+1)*N:], alpha=alpha, color='black',
            width=width, align ='edge')

    plt.title('')
    plt.xlabel('time[s]')
    plt.ylabel('#APs (layer 2/3)')
    plt.legend()

    plt.show()



###################

path='/Users/gilikarni/Google Drive/work/TU-berlin/Capstone/Code/output_files/simple_model_pyr4_bask23_G_3_W_000015_pyr23_g_2_w_3_dev_loc.json'
# for more than one json , list the paths
# plot_freq_vs_infreq_LFP([path], 8)

data1 = get_spk_data(path)

spktime1 = data1['spktime']
spkid1 = data1['spkid']
infreq_idexes1 = data1['infreq_index']


x1,y1 = generate_data_spks(spktime1, 8)
create_plot_spks(x1,y1, infreq_idexes1)
