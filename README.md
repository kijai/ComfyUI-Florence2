# Florence2 in ComfyUI

> Florence-2 is an advanced vision foundation model that uses a prompt-based approach to handle a wide range of vision and vision-language tasks. 
Florence-2 can interpret simple text prompts to perform tasks like captioning, object detection, and segmentation. 
It leverages our FLD-5B dataset, containing 5.4 billion annotations across 126 million images, to master multi-task learning. 
The model's sequence-to-sequence architecture enables it to excel in both zero-shot and fine-tuned settings, proving to be a competitive vision foundation model.

## Installation:

- clone this repository to 'ComfyUI/custom_nodes` -folder.
Only real dependency is new enough transformers version.

![image](https://github.com/kijai/ComfyUI-Florence2/assets/40791699/4d537ac7-5490-470f-92f5-3007da7b9cc7)

Supports the following models, they are automatically downloaded to `ComfyUI/LLM`:

https://huggingface.co/microsoft/Florence-2-base

https://huggingface.co/microsoft/Florence-2-base-ft

https://huggingface.co/microsoft/Florence-2-large

https://huggingface.co/microsoft/Florence-2-large-ft
