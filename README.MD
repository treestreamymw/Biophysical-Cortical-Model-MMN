# MMN in the Auditory Cortex


This repository includes a biophysical spiking neuron model of perceptual inference, based on the predictive coding framework. The model, implemented through NetPyNE (and NEURON), incorporates various single-cell models of both excitatory and inhibitory neurons, mimicking the circuits of the auditory cortex.

It allows for the exploration of the effects that biogenetic variants (expressed via ion-channels or alteration of synaptic mechanisms) have on auditory mismatch negativity (MMN) deficits, a common biomarker for schizophrenia. More specifically, the model distinguishes between repetition suppression and prediction error and examines their respective contribution to the MMN.

**A full manuscript coming soon**

## Getting Started

*RECCOMENDED:*
Install [Docker](https://docs.docker.com/install/) and use [this](https://github.com/karnigili/Computational-Neuroscience/tree/master/docker/docker_python) docker container.

*Otherwise*, install NEURON (see the specific version below) and run locally.



### An Overview of the Model

![Model overview](https://github.com/karnigili/Biophysical-Crotical-Model-MMN/blob/master/model_scheme.png)


### An Overview of the Experimental Design

1. Classic Oddball - Oddball introduces repetitive (standard) tone, which is replaced randomly by a different (deviant) tone with a low probability. We expect to see SSA to the standard tone, thus higher response to the deviant one. SSA is a property explained both by RS and MMN. This experiment helps set the baseline.
2. Flip-Flop Oddball - flipflop presents two oddball sequences with the roles of deviant and standard sounds reversed. This experiment helps control for the physical differences between the neurons associated with deviant to those associated with standard and ascribe the phenomena to adaptation rather than physiology.  
3. Many-Standards- presents a sequence of random tones of which one is uniquely equal to the used deviant above. This experiment helps us control the tone itself being unique by creating a simulation that will not evoke adaptation.  
4. Ascending Cascade- Tones are presented in a regular sequence of ascending frequencies over consecutive tones in an organized pattern.
  (a) Pattern: the pattern should not trigger adaptation since no specific tone in repeating neither MMN since the pattern is not breaking
  (b) Pattern break: An irregular tone to the pattern will trigger an MMN response (an interruption to a pattern) but will not result in an RS response.
5. Omitted Oddball- This paradigm presents a repetitive (standard) tone, which is replaced randomly by a lack of stimulus with a low probability. A lack of response to a lack of stimulus is expected in the adaptation framework. However, a response will demonstrate the neural activity to an absence of stimulus rather than an adaptation, supporting MMN.


## What Do We Have Here

### Simulation Files

* `net_params.py` includes NEURON network parameters and the simulation configuration.
* `run_sim.py` the file running the simulation based on the


### Directories
* Cells/ .hoc files configuring the different cells
* Synapses/ .mod file configuring the kinetic function of the synapses
* mod_files/  .mod files with ion channels configurations
* x86_64/ compiles .mod files (mostly to C)
* output_files/ includes plots and json of previous runs
* single_cell / includes code for running a single cell model


### Util files

* `figure_utils.py` has plotting tools
* `MP_class.py` has multiprocessing tools
* `variation_utils.py` includes tools to introduce genetic modifications to the model
* `stimulus_utils.py` includes the class that handles the stimulus (converting sensory input from a frequency to an action potential in the correct area of the cortex)

* `config.py` - includes some basic model configurations, i.e., size of the network, length, etc.

### Logging
* `Simulation_log.md` tracks the parameter variation and links to the directory holding their outputs.

## How To Run

### Run The Experiment
Run an experiment with the `run_sim.py` passing the parameters via the command-line interface.

`mpiexec -n {N} python3 run_sim.py [task] [type] [oddball] [seed] [gaba weight]`

- *Task* reflects the type of stimuli presented to the model. Options are 'oddball', 'flipflop', 'cascade', 'oddball_cascade', 'many_standards', 'no_oddball', 'omission'. See more details on the options in `stimulus_utils.py`. The default parameter is 'oddball'.
- *Type* is the network size. Options are 'full', 'short', 'mini', 'short_large'. See more details on the options in `config.py`. The default is 'full'.
- *Oddball* indicates the index of the oddball stimulus. Any integer within the configured number of pulses is acceptable. The default is 3.
- *Seed* indicates the randomization seed for the specific simulation. Any integer is acceptable. The default is 1.
- *GABA weight* reflects the percentage of GABA activity. 0 indicates no GABA, and 1 means neurotypical GABA participation. The default is 1.

### Run Analysis
Use `figure_utils.py` to generate the figures (see examples below).

#### Example

See the results of running three randomized experiments of a full simulation with a classic oddball task [here](https://github.com/karnigili/Biophysical-Crotical-Model-MMN/tree/master/output_files_sample/classic_oddball)


## Requirements
- [mpich-3.3.1](http://www.mpich.org/static/downloads/3.3.1/mpich-3.3.1.tar.gz)
- [iv-19](https://neuron.yale.edu/ftp/neuron/versions/v7.6/iv-19.tar.gz)
- [nrn-7.6.2](https://neuron.yale.edu/ftp/neuron/versions/v7.6/7.6.2/nrn-7.6.2.tar.gz)
- python3

See 'requirements.txt' for python requirements.


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
