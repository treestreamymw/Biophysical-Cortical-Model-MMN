import json
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

#plt.style.use('ggplot')


def prepare_data(LFP_dict, N_peaks, N_msrmnt, infreq_index):

    # avg 0,1 electrodes
    avg_LFP = [np.mean(i) for i in LFP_dict]

    #create a matrix and populate with the LFP data by peak
    LFP_peak_matrix = np.zeros(shape=(N_peaks,N_msrmnt))
    for i in range(N_peaks):
        LFP_peak_matrix[i]=avg_LFP[i*N_msrmnt:(i+1)*N_msrmnt]

    # seperate freq vs infreq peaks
    peak_freq = np.delete(LFP_peak_matrix, infreq_index, 0)
    peak_freq_mean = np.mean(peak_freq, axis=0)

    peak_infreq = LFP_peak_matrix[infreq_index]

    return {'infreq':peak_infreq, 'freq':peak_freq_mean}

def exctract_data(file_names_list, N_peaks, N_msrmnt):

    all_infreq_LFPs = []
    all_freq_LFP = []

    # import all files LFP data
    for file_name in file_names_list:
        with open(file_name) as f:
            data = f.read()

        j_data = json.loads(data)
        LFP = j_data['simData']['LFP']

        freq_stim = j_data['net']['params']['connParams']['Stim0->PYR4']['postConds']['x']

        for stim in range(10):
            current_stim =  j_data['net']['params']['connParams']['Stim{}->PYR4'.format(stim)]['postConds']['x']

            if current_stim != freq_stim:
                infreq_index = stim
                break

        prepared_data = prepare_data(LFP, N_peaks, N_msrmnt, infreq_index)

        all_infreq_LFPs.append(prepared_data['infreq'])
        all_freq_LFP.append(prepared_data['freq'])

    mean_infreq_LFPs = np.mean(all_infreq_LFPs, axis=0)
    mean_freq_LFP = np.mean(all_freq_LFP, axis=0)

    return {'infreq':mean_infreq_LFPs, 'freq':mean_freq_LFP}


def create_plot (data):
    plt.plot(-1*data['freq'], label='frequent', c='black')
    plt.plot(-1*data['infreq'], label='infrequent', c='coral')
    plt.title('Frequent vs Infrequent mean potentials')
    plt.legend()
    plt.show()
