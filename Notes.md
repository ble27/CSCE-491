# Model architecture

## Transformer 
* Direct classification
* Look at image and attempt to classify pixel-wise
* Example: Directly circling the boundary of an organ or color in a tumor from an scan

## Generative 
* Learn from behaviors, ground-truths, masks 
* Create new images and output new data from training 
* Example: Generate a new CT scan from an X-ray

<br><br>

# Quantization
Core idea: Affine transformation (Formulas) -> Hardware (Methods) -> Quality (Granurality) -> Strategy (PTQ vs. QAT).


## Formulas
* Quantization: $q = \text{clamp}(\lfloor{\frac{x}{S}}\rfloor + Z, Q_{min}, Q_{max})$
* Dequantization: $\hat{q} = S * (q - Z)$

<br><br>
## INT8 quantization methods
| Type | Weights only | Full INT8  |
| :---: | :---: | :---: |
| Core mechanism | Dequantize INT8 weights with FP16 scale to get FP16 weights  | Convert both weights and activation to INT8 |
| INT8 stored | Weights only (VRAM) | Both weights and activation |
| Memory footprint | Reduced by ~50% | Reduced by ~50% |
| Math | FP16/BF16 | INT8 |
| Affected by | Outlier activations (squash small weights) | Outlier activations |
| Bottleneck fixed | Memory bandwidth <br>(Retrieve INT8 instead of FP16 weights from VRAM)<br> | Computational bound <br>(Faster calculation speed because of INT8)<br>|

**Note**: Activations here refer to the intermediate output passed from 1 layer to another layer, and not the activation functions.

<br><br>
## Granularity
What is the scope of where the scaling factor (S) and zero (Z) should be applied?

| Type | Mechanism | Standard for | Quality | Overhead |
| :---: | :---: | :---: | :---: | :---: |
| Per-tensor | One scale for the network | Baseline | Low | Low |
| Per-channel | One scale for each output neuron (channel) | INT8 | High | Medium
| Group-wise | One scale for each block within a larger block | INT4 | Highest | Highest |


<br><br>
## Quantization strategies
PTQ works for most cases.<br> 
QAT works the edge cases where quantization is needed for edge devices or vision models.

### PTQ (Post-training quantization)
* Trained model -> Calibration -> Calculate scales -> Quantization -> Convert weights -> Quantized model
* Fast & Efficient
* Data privacy (Don't need full training dataset)
* Best for LLMs and Generative models

### QAT (Quantization-aware training)
* Add fake quantization during fine-tuning where the model rounds to a quantized format and convert to original format to add noise
* Need full training dataset
* Slow 
* Best for Vision models, Edge, INT4