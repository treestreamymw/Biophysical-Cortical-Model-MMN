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
import seaborn as sns
from scipy import stats
from matplotlib.transforms import Affine2D

plt.style.use('ggplot')


#### UTILS ####

def r2(x, y):
    # pearson'r R2
    return stats.pearsonr(x, y)


def get_gid_from_pop(data, pop):
    '''get cell ids from population '''
    cellGids=[]
    # if not defined, get all
    if pop is None:
        return data['simData']['spkid']

    # otherwise, find the cells of the desired population
    for cell in data['net']['cells']:
        if cell['tags']['pop'] in pop:
            cellGids.append(cell['gid'])

    return cellGids

def get_spk_data_for_pop(data, pop):
    '''
    retrieve spikes from the given populations
    '''
    cellGids=get_gid_from_pop(data,pop)
    df = pd.DataFrame(pandaslib.to_object_array([data['simData']['spkt'],\
            data['simData']['spkid']]).transpose(), \
            columns=['spkt', 'cellid'])

    spike_data = df.query('cellid in @cellGids')
    spike_data['spkind']=spike_data.index.copy()

    return spike_data

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

def find_infreq_index(j_data, file_name):
    infreq_indexes =[]
    #names = j_data['net']['params']['connParams']
    #infreq_stims = [s for s in names.keys() if "dev" in s]
    #if infreq_stims==[]:
        #if can not find, get it from file name
    infreq_indexes=[int(file_name[-13:-12])]
    #else:
        #for stim in infreq_stims:
            #infreq_indexes.append(int(stim[9]))
    return infreq_indexes

def open_file_as_json(name):
    # from json path to array
    with open(name) as f:
        data = f.read()

    return json.loads(data)

#### ANALYSIS ####
def prepare_LFP_dict(LFP_dict, N_stim, infreq_index, ms_to_trim=50, mean=True):
    '''
    extract LFP data, avged electrodes
    distinguishes frequnt to infrequent stimuli
    '''
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

    return {'infreq':peak_infreq,
        'freq':peak_freq_mean}

def get_mean_LFP_from_list(file_names_list, N_stim, trim=1.5):
    '''
    returns the avg LFP info from a list of files
    '''
    all_infreq_LFP = []
    all_freq_LFP = []

    infreq_CI = []
    max_values = []

    # import all files LFP data
    for file_name in file_names_list:

        j_data = open_file_as_json(file_name)
        LFP = j_data['simData']['LFP']

        infreq_stim = find_infreq_index(j_data, file_name)
        if infreq_stim==[]:
            infreq_stim=None
        else:
            infreq_stim=infreq_stim[0]


        prepared_data = prepare_LFP_dict(LFP, N_stim, infreq_stim, trim, True)

        all_infreq_LFP.append(prepared_data['infreq'])
        all_freq_LFP.append(prepared_data['freq'])

        max_values.append(np.max(prepared_data['infreq']))


    mean_infreq_LFP = np.mean(all_infreq_LFP, axis=0)
    mean_freq_LFP = np.mean(all_freq_LFP, axis=0)

    infreq_CI = [np.percentile(max_values, 2.5, axis=0),
                    np.percentile(max_values, 97.5, axis=0)]

    return {'infreq':mean_infreq_LFP, 'freq':mean_freq_LFP,
                'max':{'mean':np.mean(max_values),
                 'CI':infreq_CI,
                 'all':max_values} }

def prepare_spiking_stats(path, N_stim, trim_ms=50, pop=None):
    '''

    prepare data for spiking stats plots
    path - to json files
    plot type - AP or NEURONS
    N peask - the expected number of stimuli
    trim_ms - the number of ms to ignore at the start of the stim
    pop - the populations to include
    k- the numebr of ms to bin
    '''

    # get data and prepare it
    data=open_file_as_json(path)
    df=get_spk_data_for_pop(data, pop)
    infreq_id = find_infreq_index(data, path)

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

    return {'X':X,
        'spike_ids_per_t':spike_ids_per_t,
        'n_rec_neuron_per_t':n_rec_neuron_per_t,
        'infreq_id':infreq_id}

def prepare_spiking_data_for_bar_plot(path_list, plot_type, N_stim, trim_ms=50, pop=None):
    '''
    counts stats for bar plot comparing std dev and ctrl (parras)
    '''
    assert plot_type == 'AP' or plot_type == 'NEURONS', \
            'choose between AP and NEURONS'

    infreq_data=[]

    for path in path_list:
        data=prepare_spiking_stats(path=path,
                    N_stim=N_stim,
                    trim_ms=trim_ms,
                    pop=pop)

        if plot_type=='AP':
            spiking_data=data['spike_ids_per_t']
        elif plot_type=='NEURONS':
            spiking_data=data['n_rec_neuron_per_t']

        infreq_id=data['infreq_id']

        infreq_data.append(np.max(spiking_data[infreq_id]))

    infreq_mean = np.mean(infreq_data)

    infreq_ci = [np.percentile(infreq_data, 2.5, axis=0),
                    np.percentile(infreq_data, 97.5, axis=0)]

    return {'mean':infreq_mean, 'CI':infreq_ci, 'all':infreq_data}

#### PLOTS ####
def plot_freq_vs_infreq_LFP(PATH_LIST, N_stim, Raw=False):
    '''
    LFP wave of mean freq vs infreq stimuli
    '''
    if Raw:
        trim=0
    else:
        trim=2

    data = get_mean_LFP_from_list(PATH_LIST, N_stim, trim)
    stim_set=5000-500 ## 5000 reach the auditory cortex, 50ms delay from ear
    T=np.linspace(-0.05,0.3,350)

    delta= [infreq-freq for infreq,freq in zip(
        1000*data['infreq'][stim_set-500:stim_set+3000:10],
        1000*data['freq'][stim_set-500:stim_set+3000:10])]

    plt.plot(T, 1000*data['freq'][stim_set-500:stim_set+3000:10] ,
        label='frequent', c='grey', alpha=0.7)
    plt.plot(T, 1000*data['infreq'][stim_set-500:stim_set+3000:10] ,
        label='infrequent', c='coral', alpha=0.7)
    #plt.plot(T, delta ,label='MMN', c='royalblue')


    plt.axvline(x=0, label='Stimulus onset', c='cadetblue')
    plt.xlabel(' T (s)')
    plt.ylabel('Amplitude (mv)')
    ax = plt.gca()
    ax.invert_yaxis()
    plt.legend()

    if Raw:

        plt.title('Frequent vs Infrequent mean potentials')
        plt.savefig('{}/{}.png'.format(FIG_DIR_NAME,'freq_infreq_LFPs_RAW'))

    else:

        plt.title('Frequent vs Infrequent mean potentials')
        plt.savefig('{}/{}.png'.format(FIG_DIR_NAME,'freq_infreq_LFPs'))


    plt.show()
    print ('LFP plot saved')

def plot_full_LFP(file_names_list, N_stim):
    N_msrmnt = 10000

    for file_name in file_names_list:
        j_data = open_file_as_json(file_name)
        infreq_stim = find_infreq_index(j_data, file_name)
        LFP = j_data['simData']['LFP']
        ms_to_trim=5
        prepared_data = prepare_LFP_dict(LFP, N_stim, infreq_stim[0], ms_to_trim ,False)

        all_infreq_LFP=prepared_data['infreq']
        all_freq_LFP=prepared_data['freq']

        #avg_LFP = [np.mean(i) for i in LFP]
        all_peaks=[]
        for peak in range(N_stim-1):
            if peak==infreq_stim[0]:
                all_peaks.append(all_infreq_LFP)
                all_peaks.append(1000*[all_infreq_LFP[-1]])
                all_peaks.append(all_freq_LFP[peak])
            else:
                all_peaks.append(all_freq_LFP[peak])
            all_peaks.append(1000*[all_freq_LFP[peak][-1]])

        flat_peaks= [-1*item for sublist in all_peaks for item in sublist]

        N_per_stim=len(all_infreq_LFP)+1000

        #plt.plot(all_infreq_LFP, c='black')
        plt.plot(flat_peaks, c='grey')
        plt.plot(range(infreq_stim[0]*N_per_stim,(infreq_stim[0]+1)*N_per_stim),
                            flat_peaks[infreq_stim[0]*N_per_stim:(infreq_stim[0]+1)*N_per_stim],
                            c='coral')
        plt.title('LFP')
        plt.savefig('{}/{}.png'.format(FIG_DIR_NAME,'full_LFP'))
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

    data=prepare_spiking_stats(path=path,
                N_stim=N_stim,
                trim_ms=trim_ms,
                pop=pop)

    X=data['X']
    spike_ids_per_t=data['spike_ids_per_t']
    n_rec_neuron_per_t=data['n_rec_neuron_per_t']
    infreq_id=data['infreq_id']

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


    plt.title('{}, Populations : {}'.format(plot_title[plot_type],pop or 'All'))
    plt.legend()
    plt.savefig('{}/{}_{}.png'.format(FIG_DIR_NAME,plot_title[plot_type],pop))
    plt.show()
    print ('{} plot saved'.format(plot_title[plot_type]))

def plot_parras_bars(dev_path, ctrl_path, std_path, N_stim, measurement, trim=50, pop=None):
    '''
    follow parras et al (2017) figures.

    Using one simulation
    dev- is the infrequent stimulus
    control is the first stimulus
    and standard is the stimulus before the infrequent

    Reflects a bar graph comparing control, standard, and devinat across X
    simulations.
    the graph counts either the spikes count, the LFP, or the action potentials
    (based on given param).
    '''

    def get_data(path_list, N_stim, measurement, trim, pop):

        if measurement=='LFP':
            data = get_mean_LFP_from_list(path_list, N_stim, 0)
            mean = data['max']['mean']
            ci = [np.abs(i-mean) for i in data['max']['CI']]
        else:
            data=prepare_spiking_data_for_bar_plot(path_list,
                measurement, N_stim,
                    trim, pop)
            mean = data['mean']
            ci = [np.abs(i-mean) for i in data['CI']]

        return {'mean':mean, 'ci':ci}

    ## get dev data
    dev_data=get_data(dev_path, N_stim, measurement, trim, pop)
    ## get ctrl data
    ctrl_data=get_data(ctrl_path, N_stim, measurement, trim, pop)
    ## get std data
    std_data=get_data(std_path, N_stim, measurement, trim, pop)

    err = [[d,c,s] for d,c,s
            in zip(dev_data['ci'], ctrl_data['ci'], std_data['ci'])]
    data_to_plot=[dev_data['mean'], ctrl_data['mean'],std_data['mean']]
    plt.bar([1,2,3],data_to_plot, .5,
            color=['red','green','blue'],
            yerr=err)
    plt.xticks([1,2,3], ('Deviant','Control', 'Standard'))
    plt.title(measurement)

    for i in range(3):
        plt.annotate(round(data_to_plot[i],5),xy=[i+1, .05*data_to_plot[i]])

    plt.ylabel('{}'.format(measurement))
    plt.savefig('{}/{}.png'.format(FIG_DIR_NAME,
        'new_parras_bars_{}_{}'.format(measurement, pop)))

    plt.show()

def plot_A_vs_B(path_A, path_B, name_A, name_B, N_stim):
    trim=0

    data_A = get_mean_LFP_from_list(path_A, N_stim, trim)
    data_B = get_mean_LFP_from_list(path_B, N_stim, trim)

    stim_set=5000-500 ## 5000 reach the auditory cortex, 50ms delay from ear
    T=np.linspace(-0.05,0.3,350)

    MMN_A = [infreq-freq for infreq,freq in zip(
        1000*data_A['infreq'][stim_set-500:stim_set+3000:10],
        1000*data_A['freq'][stim_set-500:stim_set+3000:10])]

    MMN_B = [infreq-freq for infreq,freq in zip(
        1000*data_B['infreq'][stim_set-500:stim_set+3000:10],
        1000*data_B['freq'][stim_set-500:stim_set+3000:10])]


    plt.plot(T, MMN_A,label=name_A, c='coral')
    plt.plot(T, MMN_B, label=name_B, c='cadetblue')

    plt.title('MMN: {} vs {}'.format(name_A,name_B))
    plt.xlabel(' T (s)')
    plt.ylabel(' delta in Amplitude (mv)')


    plt.legend()
    plt.savefig('{}/{}.png'.format(FIG_DIR_NAME,'A_vs_B'))
    plt.show()

def plot_A_vs_B_bars(path_A, path_B, name_A, name_B, N_stim):
    '''
    compare 2 simulations
    '''

    data_A = get_mean_LFP_from_list(path_A, N_stim, 0)
    mean_A = data_A['max']['mean']
    ci_A = [np.abs(i-mean_A) for i in data_A['max']['CI']]
    all_A = data_A['max']['all']

    data_B = get_mean_LFP_from_list(path_B, N_stim, 0)
    mean_B = data_B['max']['mean']
    ci_B = [np.abs(i-mean_B) for i in data_B['max']['CI']]
    all_B = data_B['max']['all']

    _,t=stats.ttest_ind(all_A, all_B)

    err = [[A,B] for A,B
            in zip(ci_A, ci_B)]

    data_to_plot=[mean_A, mean_B]
    plt.bar([1,2],data_to_plot, .5,
            color=['cadetblue','forestgreen'],
            yerr=err, label=r'$p-val\leq {}$'.format(round(t,2)))
    plt.legend()
    plt.xticks([1,2], (name_A, name_B))
    plt.title('Comapring MMN between earlier to later oddball')

    for i in range(2):
        plt.annotate(round(data_to_plot[i],5),xy=[i+1, .05*data_to_plot[i]])

    plt.ylabel('{}'.format('LFP'))
    plt.savefig('{}/{}.png'.format(FIG_DIR_NAME,'comparing_bars_index'))
    plt.show()

def plot_corr_LFP(paths, independent_values, N_stim, type):
    '''
    draw correlations given a few samples
    '''
    all_means=[]
    all_cis=[]
    all_indexes=[]
    all_data_points=[]

    if type=='index':
        title='MMN as a function of oddball index'
        xlabel='Oddball index'
        filename='corr_plot_index_mmn'
    elif type=='GABA':
        title='MMN as a function of GABA level'
        xlabel='GABA level'
        filename='corr_plot_gaba_mmn'


    for path,i in zip(paths,independent_values):
        data=get_mean_LFP_from_list(path, N_stim, 0)
        data_points=data['max']['all']
        mean=data['max']['mean']
        L=len(data_points)

        all_means.append(mean)
        all_data_points.append(data_points)
        all_indexes.append(L*[i])
        all_cis.append([np.abs(i-mean) for i in data['max']['CI']])

    all_cis=np.transpose(all_cis)
    #plt.scatter(independent_values, all_means, marker='o')
    x=[item for sublist in all_indexes for item in sublist]
    y=[item for sublist in all_data_points for item in sublist]
    r,p=r2(x,y)
    fig=sns.regplot(x, y, x_estimator=np.mean,
        color ='cadetblue', label=r'$R^2={}, p-val\leq{}$'.format(round(r,2), max(round(p,2),0.05)))
    #sns.jointplot(x, y, kind="reg", stat_func=r2)

    plt.title(title)
    #plt.errorbar(independent_values, all_means, yerr=all_cis, fmt='o',linestyle='dotted')
    #plt.scatter([item for sublist in all_indexes for item in sublist],
        #[item for sublist in all_data_points for item in sublist])

    plt.ylabel('max deviant LFP')
    plt.xlabel(xlabel)
    plt.legend()
    plt.savefig('{}/{}.png'.format(FIG_DIR_NAME,filename))
    plt.show(fig)

def calc_irs_ipe(dev_path, ctrl_path, std_path, N_stim, measurement, trim=50, pop=None):
    '''
    return irs and ipe for given set of simulations
    '''

    def get_data(path_list, N_stim, measurement, trim, pop):

        if measurement=='LFP':
            data = get_mean_LFP_from_list(path_list, N_stim, 0)
            #mean = data['max']['mean']
            data_all = data['max']['all']
            #ci = [np.abs(i-mean) for i in data['max']['CI']]
        else:
            data=prepare_spiking_data_for_bar_plot(path_list,
                measurement, N_stim,
                    trim, pop)
            #mean = data['mean']
            data_all = data['all']
            #ci = [np.abs(i-mean) for i in data['CI']]

        #return {'mean':mean, 'ci':ci, 'all':data_all}
        return {'all':data_all}

    ## get dev data

    dev_data=get_data(dev_path, N_stim, measurement, trim, pop)
    ## get ctrl data
    ctrl_data=get_data(ctrl_path, N_stim, measurement, trim, pop)
    ## get std data
    std_data=get_data(std_path, N_stim, measurement, trim, pop)

    irs_all=[c-s for c,s in zip(ctrl_data['all'],std_data['all'])]
    ipe_all=[d-c for d,c in zip(dev_data['all'],ctrl_data['all'])]
    imm_all=[d-s for d,s in zip(dev_data['all'],std_data['all'])]


    irs=np.mean(irs_all)
    ipe=np.mean(ipe_all)
    imm=np.mean(imm_all)

    irs_err = [np.percentile(irs_all, 2.5, axis=0)-irs,
                    irs-np.percentile(irs_all, 97.5, axis=0)]
    ipe_err = [np.percentile(ipe_all, 2.5, axis=0)-ipe,
                    ipe-np.percentile(ipe_all, 97.5, axis=0)]
    imm_err = [np.percentile(imm_all, 2.5, axis=0)-imm,
                    imm-np.percentile(imm_all, 97.5, axis=0)]


    return {'irs':irs, 'irs_ci':irs_err,
                'ipe':ipe, 'ipe_ci':ipe_err,
                'imm':imm, 'imm_ci':imm_err }

def plot_irs_ipe(data, paradigm, measurement):
    '''
    plt irs and ipe error plots og three gaba levels
    '''

    gaba_0_5=data[0]
    gaba_0_7=data[1]
    gaba_1=data[2]

    irs=[gaba_0_5['irs'],gaba_0_7['irs'],gaba_1['irs']]
    irs_err=[gaba_0_5['irs_ci'],gaba_0_7['irs_ci'],gaba_1['irs_ci']]
    irs_err=np.transpose(irs_err)

    ipe=[gaba_0_5['ipe'],gaba_0_7['ipe'],gaba_1['ipe']]
    ipe_err=[gaba_0_5['ipe_ci'],gaba_0_7['ipe_ci'],gaba_1['ipe_ci']]
    ipe_err=np.transpose(ipe_err)

    imm=[gaba_0_5['imm'],gaba_0_7['imm'],gaba_1['imm']]
    imm_err=[gaba_0_5['imm_ci'],gaba_0_7['imm_ci'],gaba_1['imm_ci']]
    imm_err=np.transpose(imm_err)

    X=["50%","70%","100%"]
    fig, ax = plt.subplots()

    trans_mm = Affine2D().translate(-0.15, 0.0) + ax.transData
    trans_pe = Affine2D().translate(-0.05, 0.0) + ax.transData
    trans_rs = Affine2D().translate(+0.05, 0.0) + ax.transData

    plt.errorbar(X, imm, yerr=imm_err, fmt='o',linestyle="none",
        transform=trans_mm, capsize=3, capthick=1, label='iMM')

    plt.errorbar(X, irs, yerr=irs_err, fmt='o',linestyle="none",
        transform=trans_rs, capsize=3, capthick=1, label='iRS')
    data_irs = {
        'x': X,
        'y1': [y - e for y, e in zip(irs, irs_err[0])],
        'y2': [y + e for y, e in zip(irs, irs_err[1])]}

    #plt.fill_between(**data_irs, alpha=.25, transform=trans1)

    plt.errorbar(X, ipe, yerr=ipe_err, fmt='o',linestyle="none",
        transform=trans_pe, capsize=3, capthick=1, label='iPE')
    data_ipe = {
        'x': X,
        'y1': [y - e for y, e in zip(ipe, ipe_err[0])],
        'y2': [y + e for y, e in zip(ipe, ipe_err[1])]}

    #plt.fill_between(**data_ipe, alpha=.25, transform=trans2)


    plt.ylabel(measurement)
    plt.xlabel('GABA levels (as percent of original weight)')
    plt.title('{} delta as a fucntion of GABA levels : {} paradigm'.format(measurement,paradigm))
    plt.legend()
    plt.savefig('{}/{}_{}.png'.format(FIG_DIR_NAME,measurement,paradigm))
    plt.show()

def plot_irs_ipe_DOUBLE(data, paradigm, measurement, names):
    '''
    plt irs and ipe error plots og three gaba levels
    '''

    DATA1=data[0]
    DATA2=data[1]

    irs=[DATA1['irs'],DATA2['irs']]
    irs_err=[DATA1['irs_ci'],DATA2['irs_ci']]
    irs_err=np.transpose(irs_err)

    ipe=[DATA1['ipe'],DATA2['ipe']]
    ipe_err=[DATA1['ipe_ci'],DATA2['ipe_ci']]
    ipe_err=np.transpose(ipe_err)

    imm=[DATA1['imm'],DATA2['imm']]
    imm_err=[DATA1['imm_ci'],DATA2['imm_ci']]
    imm_err=np.transpose(imm_err)

    X=names
    fig, ax = plt.subplots()

    trans_mm = Affine2D().translate(-0.15, 0.0) + ax.transData
    trans_pe = Affine2D().translate(-0.05, 0.0) + ax.transData
    trans_rs = Affine2D().translate(+0.05, 0.0) + ax.transData

    plt.errorbar(X, imm, yerr=imm_err, fmt='o',linestyle="none",
        transform=trans_mm, capsize=3, capthick=1, label='iMM')

    plt.errorbar(X, irs, yerr=irs_err, fmt='o',linestyle="none",
        transform=trans_rs, capsize=3, capthick=1, label='iRS')
    data_irs = {
        'x': X,
        'y1': [y - e for y, e in zip(irs, irs_err[0])],
        'y2': [y + e for y, e in zip(irs, irs_err[1])]}

    #plt.fill_between(**data_irs, alpha=.25, transform=trans1)

    plt.errorbar(X, ipe, yerr=ipe_err, fmt='o',linestyle="none",
        transform=trans_pe, capsize=3, capthick=1, label='iPE')
    data_ipe = {
        'x': X,
        'y1': [y - e for y, e in zip(ipe, ipe_err[0])],
        'y2': [y + e for y, e in zip(ipe, ipe_err[1])]}

    #plt.fill_between(**data_ipe, alpha=.25, transform=trans2)


    plt.ylabel(measurement)
    plt.title('{} delta : {} paradigm'.format(measurement,paradigm))
    plt.legend()
    plt.savefig('{}/{}_{}.png'.format(FIG_DIR_NAME,measurement,paradigm))
    plt.show()

def plot_A_vs_B_vs_C(path_A, path_B, path_C, name_A, name_B, name_C, N_stim):
    trim=0

    data_A = get_mean_LFP_from_list(path_A, N_stim, trim)
    data_B = get_mean_LFP_from_list(path_B, N_stim, trim)
    data_C = get_mean_LFP_from_list(path_C, N_stim, trim)


    stim_set=5000-500 ## 5000 reach the auditory cortex, 50ms delay from ear
    T=np.linspace(-0.05,0.3,350)

    MMN_A = [infreq-freq for infreq,freq in zip(
        1000*data_A['infreq'][stim_set-500:stim_set+3000:10],
        1000*data_A['freq'][stim_set-500:stim_set+3000:10])]

    MMN_B = [infreq-freq for infreq,freq in zip(
        1000*data_B['infreq'][stim_set-500:stim_set+3000:10],
        1000*data_B['freq'][stim_set-500:stim_set+3000:10])]

    MMN_C = [infreq-freq for infreq,freq in zip(
        1000*data_C['infreq'][stim_set-500:stim_set+3000:10],
        1000*data_C['freq'][stim_set-500:stim_set+3000:10])]

    fig, ax = plt.subplots(2, 3)

    ax[0,0].title(name_A)
    ax[0,0].plot(T, MMN_A,label=name_A)
    ax[0,1].title(name_B)
    ax[0,1].plot(T, MMN_B, label=name_B)
    ax[0,2].title(name_C)
    ax[0,2].plot(T, MMN_C, label=name_C)

    ax[1,0].plot(T, 1000*data_A['freq'][stim_set-500:stim_set+3000:10],label=name_A)
    ax[1,0].plot(T, 1000*data_A['infreq'][stim_set-500:stim_set+3000:10],label=name_A)

    ax[1,1].plot(T, 1000*data_B['freq'][stim_set-500:stim_set+3000:10], label=name_B)
    ax[1,1].plot(T, 1000*data_B['infreq'][stim_set-500:stim_set+3000:10],label=name_B)

    ax[1,2].plot(T, 1000*data_C['freq'][stim_set-500:stim_set+3000:10], label=name_C)
    ax[1,2].plot(T, 1000*data_C['infreq'][stim_set-500:stim_set+3000:10],label=name_C)



    plt.title('MMN: {}, {}, and {}'.format(name_A, name_B, name_C))
    plt.xlabel(' T (s)')
    plt.ylabel(' delta in Amplitude (mv)')


    plt.legend()
    plt.savefig('{}/{}.png'.format(FIG_DIR_NAME,'A_vs_B_vs_C'))
    plt.show()
###################

### DATA COLLECTION ###
### oddball neurotypical
DEV_LIST=glob('output_files/experiments/NeuroTypical/classic_oddball/*.json') # oddball
CTRL_LIST=glob('output_files/experiments/NeuroTypical/many_standards/*.json') # ms
#CTRL_W_MEM=['output_files/experiments/run2/many_standards/beta_many_standards_3_seed_8.json']
STD_LIST=glob('output_files/experiments/NeuroTypical/no_oddball/*.json') # no oddball

#DEV_LIST_2=glob('output_files/experiments/gaba_alteration/no_oddball_gaba_.5/*.json') # oddball
DEV_LIST_2=glob('output_files/experiments/NeuroTypical/oddball_2/*.json') # no oddball

### oddball neuroAtypical


DEV_0_5_LIST=glob('output_files/experiments/gaba_alteration/oddball_gaba_.5/*.json')
CTRL_0_5_LIST=glob('output_files/experiments/gaba_alteration/many_standards_.5/*.json') # ms
STD_0_5_LIST=glob('output_files/experiments/gaba_alteration/no_oddball_gaba_.5/*.json')

DEV_0_7_LIST=glob('output_files/experiments/gaba_alteration/oddball_gaba_.7/*.json')
CTRL_0_7_LIST=glob('output_files/experiments/gaba_alteration/many_standards_.7/*.json') # ms
STD_0_7_LIST=glob('output_files/experiments/gaba_alteration/no_oddball_gaba_.7/*.json')

### cascade neurotypical

CASCADE_DEV_LIST=glob('output_files/experiments/NeuroTypical/oddball_cascade/*.json') # oddball
CASCADE_CTRL_LIST=glob('output_files/experiments/NeuroTypical/many_standards_5/*.json') # ms
CASCADE_STD_LIST=glob('output_files/experiments/NeuroTypical/cascade/*.json') # no oddball

### cascade neuroAtypical

CASCADE_DEV_0_5_LIST=glob('output_files/experiments/gaba_alteration/oddball_cascade_gaba_.5/*.json')
CASCADE_CTRL_0_5_LIST=glob('output_files/experiments/gaba_alteration/many_standards_.5_5/*.json') # ms
CASCADE_STD_0_5_LIST=glob('output_files/experiments/gaba_alteration/cascade_gaba_.5/*.json')

CASCADE_DEV_0_7_LIST=glob('output_files/experiments/gaba_alteration/oddball_cascade_gaba_.7/*.json')
CASCADE_CTRL_0_7_LIST=glob('output_files/experiments/gaba_alteration/many_standards_.7_5/*.json') # ms
CASCADE_STD_0_7_LIST=glob('output_files/experiments/gaba_alteration/cascade_gaba_.7/*.json')

## oddball index comparing
ind_2=glob('output_files/experiments/NeuroTypical/oddball_2/*.json') # no oddball
ind_3=glob('output_files/experiments/NeuroTypical/oddball_3/*.json') # no oddball
ind_4=glob('output_files/experiments/NeuroTypical/oddball_4/*.json') # no oddball
ind_5=glob('output_files/experiments/NeuroTypical/classic_oddball/*.json') # no oddball

## oddball corr different GABA levels
oddball_gaba_0_5=glob('output_files/experiments/gaba_alteration/oddball_gaba_.5/*.json')#[:2] # no oddball
oddball_gaba_0_7=glob('output_files/experiments/gaba_alteration/oddball_gaba_.7/*.json') # no oddball
oddball_gaba_1=glob('output_files/experiments/NeuroTypical/classic_oddball/*.json')#[:2] # no oddball


## multi gene

DEV_gene=glob('output_files/experiments/genetic_single/oddball/*.json')
CTRL_gene=glob('output_files/experiments/genetic_single/many_standards/*.json')
STD_gene=glob('output_files/experiments/genetic_single/no_oddball/*.json')


DEV_combo_gene=glob('output_files/experiments/genetic_combo/oddball/*.json')
CTRL_combo_gene=glob('output_files/experiments/genetic_combo/many_standards/*.json')
STD_combo_gene=glob('output_files/experiments/genetic_combo/no_oddball/*.json')

### FIG SAVE ###
FIG_DIR_NAME='output_files/experiments/genetic_single/'


### PLOTS ###

#oddball_data_gaba1=calc_irs_ipe(DEV_LIST, CTRL_LIST, STD_LIST, 8, 'NEURONS', 50)
#oddball_data_gene=calc_irs_ipe(DEV_gene[::2], CTRL_gene[::2], STD_gene[::2], 8, 'AP', 50)
#cascade_data_gaba1=calc_irs_ipe(CASCADE_DEV_LIST, CASCADE_CTRL_LIST, CASCADE_STD_LIST, 8, 'AP', 50)

#plot_A_vs_B_vs_C(DEV_LIST, CTRL_LIST, CASCADE_DEV_LIST, 'oddball', 'many standards', 'cascade', 8)

#oddball_data_COMBO_gene=calc_irs_ipe(DEV_combo_gene, CTRL_combo_gene, STD_combo_gene, 8, 'NEURONS', 50)

#plot_freq_vs_infreq_LFP(DEV_combo_gene, 8, Raw=True)
#plot_freq_vs_infreq_LFP(DEV_gene, 8, Raw=True)


#plot_parras_bars(DEV_combo_gene, CTRL_combo_gene, STD_combo_gene, 8, 'AP', 50)#, ['PYR23', 'BASK23'])
#plot_irs_ipe_DOUBLE([oddball_data_gaba1,oddball_data_COMBO_gene],'NeuroTypical','NEURONS', ["NeuroTypical", "Combo variant"])

#plot_A_vs_B_bars(DEV_LIST, DEV_combo_gene, 'Neurotypical', 'Combo variant',8)

#plot_parras_bars(DEV_gene, CTRL_gene, STD_gene, 8, 'AP', 50)#, ['PYR23', 'BASK23'])

#plot_corr_LFP([ind_2, ind_3, ind_4, ind_5],[2,3,4,5], 8, 'index')


#oddball_data_gaba1=calc_irs_ipe(DEV_LIST, CTRL_LIST, STD_LIST, 8, 'NEURONS', 50)
#oddball_data_gaba_07=calc_irs_ipe(DEV_0_7_LIST, CTRL_0_7_LIST, STD_0_7_LIST, 8, 'NEURONS', 50)
#oddball_data_gaba_05=calc_irs_ipe(DEV_0_5_LIST, CTRL_0_5_LIST, STD_0_5_LIST, 8, 'NEURONS', 50)

#plot_freq_vs_infreq_LFP(DEV_gene, 8, Raw=True)
plot_spiking_stats_df(DEV_gene[0],'NEURONS',8, 50)


#plot_irs_ipe([oddball_data_gaba_0_5,oddball_data_gaba_07,oddball_data_gaba1],'Oddball','NEURONS')


#cascade_data_gaba1=calc_irs_ipe(CASCADE_DEV_LIST, CASCADE_CTRL_LIST, CASCADE_STD_LIST, 8, 'AP', 50)
#cascade_data_gaba_07=calc_irs_ipe(CASCADE_DEV_0_7_LIST, CASCADE_CTRL_0_7_LIST, CASCADE_STD_0_7_LIST, 8, 'AP', 50)
#cascade_data_gaba_05=calc_irs_ipe(CASCADE_DEV_0_5_LIST, CASCADE_CTRL_0_5_LIST, CASCADE_STD_0_5_LIST, 8, 'AP', 50)
#plot_irs_ipe([cascade_data_gaba_05,cascade_data_gaba_07,cascade_data_gaba1],'Casacde','AP')


#plot_corr_LFP([gaba_0_5, gaba_0_7, gaba_1],[0.5,0.7,1], 8, 'GABA')
#plot_A_vs_B_bars(DEV_LIST, DEV_LIST_2, 'High index','Low index',8)
#plot_freq_vs_infreq_LFP(oddball_gene, 8, Raw=True)
#plot_parras_bars(CASCADE_DEV_LIST, CASCADE_CTRL_LIST, CASCADE_STD_LIST, 8, 'AP', 50)#, ['PYR23', 'BASK23'])
#plot_spiking_stats_df(DEV_LIST[0],'NEURONS',8, 50, ['PYR23'])
