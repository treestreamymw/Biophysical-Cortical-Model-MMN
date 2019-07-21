# MMN in the auditory cortex

This includes a NEURON model of the auditory cortex. Inspired by the work of

‘Wacongne, C., Changeux, J. P., & Dehaene, S. (2012). A neuronal model of predictive coding accounting for the mismatch negativity. Journal of Neuroscience, 32(11), 3665-3678.’

‘Auksztulewicz, R. & Friston, K.J. (2016). Repetition suppression and its contextual determinants in predictive coding. Cortex, 80, 125-140.’

‘Bastos, A.M., Usrey, W.M., Adams, R.A., Mangun, G.R., Fries, P., Friston, K.J. (2012). Canonical microcircuits for predictive coding. Neuron, 76, 695-711.’

The model architecture is based on the canonical microcircuit for predictive coding outlined in Bastos et al. (2012) which presents a simplified scheme. The model includes of a granular (Layer 4) and supergranular (Layer 2/3). we do not include an infragranular (Layer 5/6) layer, for reasons explained below. Each includes excitatory and inhibitory population.

We model forward and backward connectivity, carrying prediction and prediction error messages respectively.



## Getting Started

Install NERUON and set your env using the docker file.


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


* config- NEED TO BE MODIFIED (includes some basic confs)



## License

This project is licensed under the MIT License


## Contributors

Christoph Metzner, Gili Karni
