# MMN in the auditory cortex



This repository includes a NEURON model of the auditory cortex.

## Getting Started

*RECCOMENDED:*
Install [Docker](https://docs.docker.com/install/) and use [this](https://github.com/karnigili/Computational-Neuroscience/tree/master/docker/docker_python) docker container.

*Otherwise*, install NEURON (see the specific version below) and run locally.


## Goal and simulation descriptions

This code simulates activation in the auditory cortex, including layer 2/3 and layer 4. The full-size network contains 18 (9) rows with 24 (12) neurons at each row (to simulate eight frequencies), and the short version includes 8 (4) rows with 24 (12) neurons at each row (to simulate two frequencies).

We aim to explore the differences between Stimulus Specific Adaptation (SSA) and MisMatch Negativity (MMN) in healthy in SZ-like cortical systems.

*A full manuscript coming soon*

### An overview of the model

The model is designed to reflect a primary auditory cortex to study the MMN phenomenon. The model architecture is based on the canonical microcircuit for predictive coding outlined in Bastos et al. (2012), which presents a simplified scheme. The model includes a granular (Layer 4) and super-granular (Layer 2/3). We do not incorporate an infra-granular (Layer 5/6) layer, for reasons explained below. Each includes an excitatory and inhibitory population.

#### The network level

The network's architecture synthesizes Kudela et al. (2018) work on SSA in the auditory cortex, and Bastos et al. (2012) work on predictive coding.

The network consists of two cortical layers; layer 2/3 and layer 4. We discount layer 5/6 since it does not significantly support MMN (Bastos et al, 2012) and would increase the computational load. An additional auxiliary layer demonstrates the auditory afferent inputs from the thalamus to layer 4.

The model represents the frequency-specific (tonotopic) organization of the auditory cortex, such that each row of neurons corresponds to a different frequency.   

Each cortical layer illustrates a population of excitatory (pyramidal) cells and inhibitory (basket) cells. Each excitatory population has 336 neurons organized in 14 rows (24 neurons per row), and each inhibitory population had 84 neurons arranged in 7 rows (12 neurons per row). Such a ratio and organization follow the biological structure of the cortex. The auxiliary layer consists of eight populations of 24 cells, corresponding with the number of stimuli presented to the model.

Following Bastos et al. (2012), layer 2/3 contains a secondary excitatory population supporting the prediction error stream (information stream from deeper regions). More details about it below.

The general network connectivity follows the canonical microcircuit for predictive coding outlined in Bastos et al. (2012).  Inter-layer connectivity features excitatory (NMDA, AMPA) and inhibitory (GABA) synapses. Inter-laminar connectivity features short term plasticity via modified AMPA synapses. All but the addition of the NMDA receptor is a replication of Kudela et al., (2018). The choice to ass NMDA receptors stemmed from their importance in SZ and MMN.

#### The cellular level
Simulated neurons were biophysically realistic
multi compartmental cell models.

Cortical excitatory cells simulated here via a Hay pyramidal cell model. The cells of layer 4 incorporate a simple four compartments cell, and the excitatory of layer 2/3 present a more complex model.

- Layer 4 Pyramidal cell (fourcompartment).
- Layer23 Pyramidal cell (pyr_23_asym_stripped)
- Memory layer Pyramidal cell (pyr_23_asym_stripped)


The inhibitory cells of both layers are simple two compartments basket cells.

- Basket cell (FS)

#### The biophysical level

We employ three types of synaptic mechanisms;
- Exp2Syn - a double exponential mechanism used in the AMPA and GABA synapses.
- FDSExp2Syn- a variation of Exp2Syn that demonstrates fast facilitation mechanism. Implemented in-between layer connections.
- NMDA- unique for NMDA cells, displays a magnesium blocked channel.

### An overview of the experimental design

1. Classic Oddball - Oddball introduces repetitive (standard) tone, which is replaced randomly by a different (deviant) tone with a low probability. We expect to see SSA to the standard tone, thus higher response to the deviant one. SSA is a property explained both by RS and MMN. This experiment helps set the baseline.
2. Flip-Flop Oddball - flipflop presents two oddball sequences with the roles of deviant and standard sounds reversed. This experiment helps control for the physical differences between the neurons associated with deviant to those associated with standard and ascribe the phenomena to adaptation rather than physiology.  
3. Many-Standards- presents a sequence of random tones of which one is uniquely equal to the used deviant above. This experiment helps us control the tone itself being unique by creating a simulation that will not evoke adaptation.  
4. Ascending Cascade- Tones are presented in a regular sequence of ascending frequencies over consecutive tones in an organized pattern.
  (a) Pattern: the pattern should not trigger adaptation since no specific tone in repeating neither MMN since the pattern is not breaking
  (b) Pattern break: An irregular tone to the pattern will trigger an MMN response (an interruption to a pattern) but will not result in an RS response.
5. Omitted Oddball- This paradigm presents a repetitive (standard) tone, which is replaced randomly by a lack of stimulus with a low probability. A lack of response to a lack of stimulus is expected in the adaptation framework. However, a response will demonstrate the neural activity to an absence of stimulus rather than an adaptation, supporting MMN.



## What do we have here

### Simulation files

* `net_params.py` includes NEURON network parameters and the simulation configuration.
* `run_sim.py` the file running the simulation based on the

### Directories

* Cells/ .hoc files configuring the different cells
* Synapses/ .mod file configuring the kinetic function of the synapses
* mod_files/  .mod files with ion channels configurations
* x86_64/ compiles .mod files (mostly to C)
* output_files/ includes plots and json of previous runs
* single_cell / includes code for running a single cell model


### utils

* `figure_utils.py` has plotting tools
* `MP_class.py` has multiprocessing tools
* `variation_utils.py` includes tools to introduce genetic modifications to the model
* `stimulus_utils.py` includes the class that handles the stimulus (converting sensory input from a frequency to an action potential in the correct area of the cortex)

* `config.py` - includes some basic model configurations, i.e., size of the network, length, etc.

### Logging
* `Simulation_log.md` tracks the parameter variation and links to the directory holding their outputs.

## How to run

### Run the experiment
Run an experiment with the `run_sim.py` passing the parameters via the command-line interface.

`mpiexec -n {N} python3 run_sim.py [task] [type] [oddball] [seed] [gaba weight]`

- *Task* reflects the type of stimuli presented to the model. Options are 'oddball', 'flipflop', 'cascade', 'oddball_cascade', 'many_standards', 'no_oddball', 'omission'. See more details on the options in `stimulus_utils.py`. The default parameter is 'oddball'.
- *Type* is the network size. Options are 'full', 'short', 'mini', 'short_large'. See more details on the options in `config.py`. The default is 'full'.
- *Oddball* indicates the index of the oddball stimulus. Any integer within the configured number of pulses is acceptable. The default is 3.
- *Seed* indicates the randomization seed for the specific simulation. Any integer is acceptable. The default is 1.
- *GABA weight* reflects the percentage of GABA activity. 0 indicates no GABA, and 1 means neurotypical GABA participation. The default is 1.


## Requirements
- [mpich-3.3.1](http://www.mpich.org/static/downloads/3.3.1/mpich-3.3.1.tar.gz)
- [iv-19](https://neuron.yale.edu/ftp/neuron/versions/v7.6/iv-19.tar.gz)
- [nrn-7.6.2](https://neuron.yale.edu/ftp/neuron/versions/v7.6/7.6.2/nrn-7.6.2.tar.gz)
- python3
- mpi4py
- NetPyNE


## References

Inspired by the work of

‘Wacongne, C., Changeux, J. P., & Dehaene, S. (2012). A neuronal model of predictive coding accounting for the mismatch negativity. Journal of Neuroscience, 32(11), 3665-3678.’

‘Auksztulewicz, R. & Friston, K.J. (2016). Repetition suppression and its contextual determinants in predictive coding. Cortex, 80, 125-140.’

‘Bastos, A.M., Usrey, W.M., Adams, R.A., Mangun, G.R., Fries, P., Friston, K.J. (2012). Canonical microcircuits for predictive coding. Neuron, 76, 695-711.’

’Kudela, P., Boatman-Reich, D., Beeman, D., & Anderson, W. S. (2018). Modeling Neural Adaptation in Auditory Cortex. Frontiers in neural circuits, 12, 72.’


## License
This project is licensed under the MIT License


## Contributors
Gili Karni, Christoph Metzner
