## simulation log

*** after theoretical decision on structure

### (1) original weights
- Files are [here](../output_file/pyr4_23_orig)
- __Observation__: we see no adaptation of layer 2/3 for via a constant number of recruited neurons and number of APs
- __Chosen action__: reduce weight on pyr4_pyr23 weight
- __Hypothesis__: the EE weight will reduce spatial excitation.
- __Justification__: the intra-laminar connectivity dictates the adaptation (via short term depression). The time of depression is 9200 ms thus perhaps weaker excitation would result in stronger adapation
-

### (2) Lower intra-laminar EE 1st (pyr4_23 w = 0.03)
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

### (5) Lower intra-laminar EE & Higher inter-laminar EE 1st ( pyr4_23 w = 0.03, pyr23_23  [0.0048,0.00024])
- Files are [here](../output_file/pyr4_23_div10_pyr23_23_double)
- __Observation__: Resembles attempt (2)- very short excitation, good adaptation.
- __Chosen action__: Increase EE inter-laminar even more.
- __Justification__:  It seems that the EE was not raised enough.


### (6) Lower intra-laminar EE & Higher inter-laminar EE 2nd ( pyr4_23 w = 0.03, pyr23_23  [0.024,0.0012])
- Files are [here](../output_file/pyr4_23_div10_pyr23_23_mult10)
- __Observation__: excitation increased only spatially (no adaptation at all) but did not lead to more excitation temporally
- __Chosen action__: reduce inhibition to 0
- __Justification__: we want to check whether the excitation is insufficient to increase the temporal dimension or if the inhibition counter it.

### (7) Lower intra-laminar EE & Higher inter-laminar EE & no inter-laminar IE ( pyr4_23 w = 0.03, pyr23_23  [0.024,0.0012], bask23_pyr23 = 0)
- Files are [here](../output_file/pyr4_23_div10_pyr23_23_mul10_bask23_pyr23_0)
- __Observation__: Same as sim (6)
- __Chosen action__: go back and try another Lower intra-laminar EE (div 5)
- __Hypothesis__: Neural sharpening is the result of the interaction between pyr4 and pyr23, BUT we see that the EE in layer 23 affects the number of recruited neurons and not their temporal repeated spiking (within the range of weight modifications we tested)- also, the IE in layer 23 has no affect on either aspect of the adaptation. Thus, we go back to explore the intra laminar relations.
- __Justification__: it seems that we can not get longer activation from within the layer, thus we need to find a balanced intra-layer weight that reduce the excitation without dropping the termporal dimension completely.


### (8) Lower intra-laminar EE - 4th  (pyr4_23 w = 0.06)
- Files are [here](../output_file/pyr4_23_div3)
- __Observation__: it seem that there is not sharpening at all. the conclusions is that the long term depression is too low and the short term depression is too strong
- __Chosen action__: go back to div10 (where we did see sharpening but a very short excitation) and increase the short term depression to 100% (i.e no short term depression)
- __Justification__: based on the conclusion, we try to maintain the existing long term depression (i.e sharpening) but not to loose the temporal scope



### (10) Lower intra-laminar EE & short depression increase  (pyr4_23 w = 0.03 d1=1.0)
- Files are [here](../output_file/pyr4_23_div10_D1_1)
- __Observation__: there is less short term depression but not fully (we believe that it has to do with the inhibition)
- __Chosen action__: Run the full simulation (8 spikes) and decrease EI between 4 to 23 to maintain excitation
- __Justification__: Keep working on the intra layer connectivity to modify the adaptation


### (11) FULL- Lower intra-laminar EE  & short depression increase  (pyr4_23 w = 0.03 d1=1.0  )
- Files are [here](../output_file/pyr4_23_div10_D1_1_full)
- __Observation__: there is less short term depression but not fully (we believe that it has to do with the inhibition)
- __Chosen action__: Run the full simulation (8 spikes) # and decrease EI between 4 to 23 to maintain excitation
- __Justification__: Keep working on the intra layer connectivity to modify the adaptation


### (12) FULL- Lower intra-laminar EE and IE  & short depression increase  (pyr4_23 w = 0.03 d1=1.0  pyr4 bask23  0.000015)
- Files are [here](../output_file/pyr4_23_div10_pyr4_bask23_div_10_d1_1)
- __Observation__: The short term depression increased but not enough so I will try to remove the IE
- __Chosen action__:Same but a bit less IE
- __Justification__: Keep working on the intra layer connectivity to modify the adaptation
