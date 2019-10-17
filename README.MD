# MMN in the auditory cortex

This includes a NEURON model of the auditory cortex. Inspired by the work of

‘Wacongne, C., Changeux, J. P., & Dehaene, S. (2012). A neuronal model of predictive coding accounting for the mismatch negativity. Journal of Neuroscience, 32(11), 3665-3678.’

‘Auksztulewicz, R. & Friston, K.J. (2016). Repetition suppression and its contextual determinants in predictive coding. Cortex, 80, 125-140.’

‘Bastos, A.M., Usrey, W.M., Adams, R.A., Mangun, G.R., Fries, P., Friston, K.J. (2012). Canonical microcircuits for predictive coding. Neuron, 76, 695-711.’

The model architecture is based on the canonical microcircuit for predictive coding outlined in Bastos et al. (2012) which presents a simplified scheme. The model includes of a granular (Layer 4) and supergranular (Layer 2/3). we do not include an infragranular (Layer 5/6) layer, for reasons explained below. Each includes excitatory and inhibitory population.

We model forward and backward connectivity, carrying prediction and prediction error messages respectively.



## Getting Started

Install NERUON and set your env using the docker file.

## Goal and simulation descriptions

This code simulates activation in the auditory cortex including layer 2/3 and layer 4. The full size network contains 18 (9) rows with 24 (12) neurons at each row (to simulate 8 frequencies) and the short version includes 8 (4) rows with 24 (12) neurons at each row (to simulate 2 frequencies).

We aim to explore the differences between Stimulus Specific Adaptation (SSA) and MisMatch Negativity (MMN) in healthy in SZ-like cortical systems.

### An overview of the model


The model consists of two main layers; layer 2/3 and layer 4. Each layer illustrates a population of excitatory (pyramidal) cells and inhibitory (basket) cells. Each excitatory population has X neurons organized in X rows, and each inhibitory population had X/2 neurons organized in X rows (following biological ratio and topology). Layer 2/3 contains a secondary excitatory population. (See grid)

The excitatory cells of layer 4 incorporate a simple four compartment cell (CITE) (FIG) and the excitatory of layer 2/3 present a more complex model (Beeman) 



pyramidal cells of layer 4, basket cells of layer 4, first pyramidal cells of layer 2/3, secondary pyramidal cells of layer 2/3 basket cells of layer 2/3.

### An overview of the experimental design

1. Odd-ball sequences - Oddball introduces repetitive (standard) tone which is replaced randomly by a different (deviant) tone with a low probability. We expect to see SSA to the standard tone thus higher response to the deviant one. SSA is a property explained both by RS and MMN. This helps set the baseline.
2. Flip-flop sequences - flipflop presents two oddball sequences with the roles of deviant and standard sounds reversed. This helps control for the physical differences between the neurons associated with deviant to those associated with standard and ascribe the phenomena to adaptation rather than physiology.  
3. Many-standards - presents a sequence of random tones of which one is uniquely equal to the used deviant above. This helps us control for the tone itself being unique by creating a simulation that will not evoke adaptation.  
4. Ascending cascade sequences- Tones are presented in a regular sequence of  ascending frequencies over consecutive tones in an organized pattern.
  (a) Pattern: the pattern should not trigger adaptation, since no specific tone in repeating neither MMN since the pattern is not breaking
  (b) Pattern break: An irregular tone to the pattern will trigger an MMN response (an interruption to a pattern) but won't result in an RS response.
5. Omission- This paradigm presents a repetitive (standard) tone which is replaced randomly by a lack of stimulus with a low probability. A lack of response to a lack of stimulus is expected in the adaptation framework. Yet, a response will demonstrate neural activity to an absence of stimulus rather than an adaptation, supporting MMN.



## What do we have here

### Simulation files

* net_params.py includes NEURON network parameters and the simulation configuration.
* net_oddball_sequence.py the file running the simulation based on the

### Directories

* mod_files includes .mod files with neuron configurations
* synapses includes .mod file configuring synaptic connections

### utils

* figure_utils.py has plotting tools
* MP_class.py has multi processing tools
* variation_utils.py includes tools to introduce genetic modifications to the model


* config- NEED TO BE MODIFIED (includes some basic configurations)

## How to run

### set config for type of experiment and network
TBD

### Run the experiment
'mpiexec -n {N} python3 net_sequence.py'


## License

This project is licensed under the MIT License


## Contributors

Christoph Metzner, Gili Karni
