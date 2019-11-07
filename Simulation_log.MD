## simulation log

*** after theoretical decision on structure

### (1) original weights
- Files are [here](../output_file/pyr4_23_orig)
- __Observation__: we see no adaptation of layer 2/3 for via a constant number of recruited neurons and number of APs
- __Chosen action__: reduce weight on pyr4_pyr23 weight
- __Hypothesis__: the EE weight will reduce spatial excitation.
- __Justification__: the intra-laminar connectivity dictates the adaptation (via short term depression). The time of depression is 9200 ms thus perhaps weaker excitation would result in stronger adapation
-

### (2) Lower intra-laminar EE (pyr4_23 w = 0.03)
- Files are [here](../output_file/pyr4_23_div10)
- __Observation__: A very strong adaptation resulting from both temporal and spatial attenuation in activation
- __Chosen action__: increase the value by a little bit
- __Justification__: we still consider this hypothesis relevant and try to see how the weight affects the result (using different values)

### (3) Lower intra-laminar EE - 2nd (pyr4_23 w = 0.15)
- Files are [here](../output_file/pyr4_23_div2)
- __Observation__: A very similar picture to the original values. Not much of a difference
- __Chosen action__: reduce a little bit
- __Justification__: we still consider this hypothesis relevant and try to see how the weight affects the result (using different values)

### (4) Lower intra-laminar EE - 3rd  (pyr4_23 w = 0.1)
- Files are [here](../output_file/pyr4_23_div3)
- __Observation__: This results in a temporal adaptation but does not affect the number of recruited neurones. Now we can conclude that the affect of EE weight reduction is mostly temporal.
- __Chosen action__: increase pyr23_23 EE weight
- __Hypothesis__: local excitation will increase the temporal excitation without impacting the spatial one.
- __Justification__: local excitation will enhance already active neurons but is less likely to activate further ones.

### (5) Lower intra-laminar EE & Higher inter-laminar EE ( pyr4_23 w = 0.03, pyr23_23  [0.0048,0.00024])
- Files are [here](../output_file/pyr4_23_div10_pyr23_23_double)
- __Observation__: Resembles attempt (2)- very short excitation, good adaptation.
- __Chosen action__: Increase EE inter-laminar even more.
- __Justification__:  It seems that the EE was not raised enough.
