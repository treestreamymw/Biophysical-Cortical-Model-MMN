import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try: # Pandas 0.24 and later
    from pandas import _lib as pandaslib
except: # Pandas 0.23 and earlier
    from pandas import lib as pandaslib
from glob import glob
import os
import re
import collections
import argparse

plt.style.use('ggplot')



def get_gid_from_pop(data, pop):
    '''get cell ids from population '''
    cellGids=[]
    if pop is None:
        return data['simData']['spkid']
    for cell in data['net']['cells']:
        if cell['tags']['pop'] in pop:
            cellGids.append(cell['gid'])
    return cellGids

def get_spk_data_for_pop(data, pop):

    cellGids=get_gid_from_pop(data,pop)
    df = pd.DataFrame(pandaslib.to_object_array([data['simData']['spkt'],\
            data['simData']['spkid']]).transpose(), \
            columns=['spkt', 'cellid'])
    sel = df.query('cellid in @cellGids')
    sel['spkind']=sel.index.copy()

    return sel

def trim_and_round_time_for_spikes(df, N_stim, trim_ms=50, round_k=100):
    '''
    trims the first N ms from each spike (ignore the initial respose) and rounds
    the data to bins of size K

    df - the data
    N_stim= number of stimuli
    trim_ms- how many ms to trim
    round_k- the bin size

    '''

    df['spkt_mod']=df['spkt']
    df['spkt_mod']=df['spkt_mod'].astype(int)

    trim_list=[i for j in [range(j*1000, j*1000+trim_ms) for j in range(N_stim)] for i in j]

    trim_first_n_ms_index = df.index[df['spkt_mod'].isin(trim_list)].tolist()
    df=df.drop(trim_first_n_ms_index)
    df['spkt_mod']=df['spkt_mod'].floordiv(round_k)

    data_per_spike=df.groupby('spkt_mod').nunique()[['cellid','spkind']]
    spks=data_per_spike['spkind'].tolist()
    cells=data_per_spike['cellid'].tolist()
    t_index=data_per_spike.index.tolist()

    return {'spikes':spks, 'n_recruited_neurons':cells,'t_index':t_index}

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
def prepare_data_LFP(LFP_dict, N_stim, infreq_index, ms_to_trim=50, mean=True):

    # avg 0,1 electrodes
    avg_LFP = [np.mean(i) for i in LFP_dict]

    #create a matrix and populate with the LFP data by peak
    LFP_peak_matrix = np.zeros(shape=(N_stim,10000))

    for i in range(N_stim):
        LFP_peak_matrix[i]=avg_LFP[i*10000:(i+1)*10000]

        #remove initial peak
        #flat before the stimuli
        LFP_peak_matrix[i][:1000]=LFP_peak_matrix[i][1000]

        #remvove first response
        trimmed= int(5000 + (ms_to_trim*100))
        LFP_peak_matrix[i][5000:trimmed]=LFP_peak_matrix[i][5000]

    # seperate freq vs infreq peaks
    peak_freq = np.delete(LFP_peak_matrix, infreq_index, 0)
    if mean:
        peak_freq_mean = np.mean(peak_freq, axis=0)
    if not mean:
        peak_freq_mean=peak_freq
    if infreq_index is None:
        peak_infreq=None
    else:
        peak_infreq = LFP_peak_matrix[infreq_index]

    return {'infreq':peak_infreq, 'freq':peak_freq_mean}

def exctract_data_LFP(file_names_list, N_stim, trim=1.5):
    all_infreq_LFPs = []
    all_freq_LFP = []

    # import all files LFP data
    for file_name in file_names_list:

        j_data = open_file_as_json(file_name)
        LFP = j_data['simData']['LFP']

        infreq_stim = find_infreq_index(j_data)
        if infreq_stim==[]:
            infreq_stim=None
        else:
            infreq_stim=infreq_stim[0]


        prepared_data = prepare_data_LFP(LFP, N_stim, infreq_stim, trim, True)

        all_infreq_LFPs.append(prepared_data['infreq'])
        all_freq_LFP.append(prepared_data['freq'])

    mean_infreq_LFPs = np.mean(all_infreq_LFPs, axis=0)
    mean_freq_LFP = np.mean(all_freq_LFP, axis=0)

    return {'infreq':mean_infreq_LFPs, 'freq':mean_freq_LFP}

def plot_freq_vs_infreq_LFP (PATH_LIST, N_stim, Raw=False):
    if Raw:
        trim=0
    else:
        trim=2
    data = exctract_data_LFP(PATH_LIST, N_stim, trim)
    if not Raw:
        plt.plot(1000*data['freq'] ,label='frequent', c='grey')
        plt.plot(1000*data['infreq'] ,label='infrequent', c='coral')
        plt.title('Frequent vs Infrequent mean potentials')
        plt.xlabel(' T (s)')
        plt.ylabel('Amplitude (mv)')

        ax = plt.gca()
        ax.invert_yaxis()
        plt.legend()
        plt.savefig('output_files/{}/{}.png'.format(FIG_DIR_NAME,'freq_infreq_LFPs'))

    else:
        stim_set=5000-500 ## 5000 reach the auditory cortex, 50ms delay from ear
        T=np.linspace(-0.05,0.3,350)

        plt.plot(T, 1000*data['freq'][stim_set-500:stim_set+3000:10] ,label='frequent', c='grey')
        plt.plot(T, 1000*data['infreq'][stim_set-500:stim_set+3000:10] ,label='infrequent', c='coral')

        plt.axvline(x=0, label='Stimulus onset', c='cadetblue')
        plt.title('Frequent vs Infrequent mean potentials <Raw>')

        plt.xlabel(' T (s)')
        plt.ylabel('Amplitude (mv)')

        ax = plt.gca()
        ax.invert_yaxis()
        plt.legend()
        plt.savefig('output_files/{}/{}.png'.format(FIG_DIR_NAME,'freq_infreq_LFPs_RAW'))




    plt.show()
    print ('LFP plot saved')

def plot_full_LFP(file_names_list, N_stim):
    N_msrmnt = 10000

    for file_name in file_names_list:
        j_data = open_file_as_json(file_name)
        infreq_stim = find_infreq_index(j_data)
        LFP = j_data['simData']['LFP']
        ms_to_trim=5
        prepared_data = prepare_data_LFP(LFP, N_stim, infreq_stim[0], ms_to_trim ,False)

        all_infreq_LFPs=prepared_data['infreq']
        all_freq_LFP=prepared_data['freq']

        #avg_LFP = [np.mean(i) for i in LFP]
        all_peaks=[]
        for peak in range(N_stim-1):
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
        plt.title(' LFP  (trimmed first {} ms)'.format(ms_to_trim))
        plt.show()

def plot_spiking_stats_df(path, plot_type, N_stim, trim_ms=50, pop=None):
    '''
    path - to json files
    plot type - AP or NEURONS
    N peask - the expected number of stimuli
    trim_ms - the number of ms to ignore at the start of the stim
    pop - the populations to include
    k- the numebr of ms to bin
    '''

    assert plot_type == 'AP' or plot_type == 'NEURONS', \
        'choose between AP and NEURONS'

    plot_title= {'AP':'# of AP', 'NEURONS':'# of recruited neurons'}
    # get data and prepare it
    data=open_file_as_json(path)
    df=get_spk_data_for_pop(data, pop)
    infreq_id = find_infreq_index(data)
    if infreq_id==[]:
        infreq_id=None
    else:
        infreq_id=infreq_id[0]


    trimmed_data=trim_and_round_time_for_spikes(df, N_stim, \
            trim_ms=trim_ms, round_k=100)

    t_index=trimmed_data['t_index']
    spks=trimmed_data['spikes']
    cells=trimmed_data['n_recruited_neurons']

    # prepare dicts to collect data for plot

    spike_ids_per_t={i:[] for i in range(N_stim)}
    n_rec_neuron_per_t={i:[] for i in range(N_stim)}
    X={i:[] for i in range(N_stim)}

    prev_t=0
    i=0

    for time_ind, t in enumerate(t_index):
        curr_t=t

        if int(curr_t)//10!=int(prev_t)//10:
            i+=1

        prev_t=curr_t

        X[i].append(t)
        spike_ids_per_t[i].append(spks[time_ind])
        n_rec_neuron_per_t[i].append(cells[time_ind])

    label_ploted=False
    for stim_i in range(N_stim):
        if stim_i==infreq_id:
            c='coral'
            label='infrequent'
        else:
            c='grey'
            if label_ploted:
                label=None
            else:
                label='frequent'
                label_ploted=True

        if plot_type=='AP':
            plt.plot([X[stim_i][0]]+X[stim_i]+[X[stim_i][-1]],\
                [0]+spike_ids_per_t[stim_i]+[0], c=c, label=label)
        elif plot_type=='NEURONS':
            plt.bar(X[stim_i],n_rec_neuron_per_t[stim_i], color=c)


    plt.title('{}, Populations :{}'.format(plot_title[plot_type],pop))
    plt.legend()
    plt.savefig('output_files/{}/{}_{}.png'.format(FIG_DIR_NAME,plot_title[plot_type],pop))
    plt.show()
    print ('{} plot saved'.format(plot_title[plot_type]))



###################
if __name__ == "__main__":
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("DIR", help="enter the directory name", type=str)
    parser.add_argument("JSON", help="enter the json file name", type=str)
    args = parser.parse_args()

    FIG_DIR_NAME=args.DIR#'pyr4_23_div10'
    JSON_FILE_NAME=args.JSON#'prediction_layer_decrease_w_pyr4_pyr23.json'
    path='output_files/{}/{}'.format(FIG_DIR_NAME,JSON_FILE_NAME)
    '''

    path='output_files/random_run_beta_model_short_stim/'
    path_list=glob('output_files/random_run_beta_model/*.json')
    FIG_DIR_NAME='random_run_beta_model_short_stim'
    #plot_spiking_stats_df(path_list[0], 'AP', 8, 50, ['PYR23'])#,'PYR_prediction'])
    #plot_spiking_stats_df(path_list[0], 'NEURONS', 8, 50, ['PYR23'])#,'PYR_prediction'])
    plot_freq_vs_infreq_LFP(path_list, 8, Raw=True)
