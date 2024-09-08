# Stable Diffusion - Image to Prompts

![SD](https://github.com/user-attachments/assets/05b5635c-45fd-43a9-b391-e704590e9e3a)

## Background
The aim is to predict the prompts that were used to generate target images, and measure the performance of the model's prediction results through mean cosine similairty (predicted vs. actual prompt embedding vectors). Images were generated from the prompts using Stable Diffusion 2.0 (768-v-ema.ckpt) and were generated with 50 steps at 768x768 px and then downsized to 512x512 for the competition dataset. 

## Data
DiffusionDB: https://www.kaggle.com/datasets/alexandreteles/diffusiondb-metadata <br>
Conceptual Captions: https://huggingface.co/datasets/conceptual_captions <br>
pszemraj data: https://huggingface.co/datasets/pszemraj/text2image-multi-prompt <br>
LAION COCO: https://huggingface.co/datasets/laion/laion-coco <br>
- Generate text prompts [1] using a text pre-trained model: `prompts-generation-pipeline.ipynb` <br>
https://huggingface.co/Gustavosta/MagicPrompt-Stable-Diffusion
- Filter out similarity higher than 0.8: `stable-diffusion-prompts-sim-filter-on-faiss.ipynb`
- Generate image using [1] prompts: `sdip-gpu-pipeline.ipynb`
- Table data: <br>
`sd2_meta_211k_filter.csv` - from Diffusion DB filtering <br>
`sd2_meta_540k_part1.csv` <br>
`sd2_meta_540k_part2.csv` - pre-trained model generated from text prompts for filtering <br>
`sd2_meta_cc660k_part1.csv` <br>
`sd2_meta_cc660k_part2.csv` - from Conceptual Captions filtering <br>
`sd2_meta_hard_filter.csv` - Filter from the following dataset: <br>
https://www.kaggle.com/datasets/motono0223/gustavosta-stable-diffusion-prompts-sd2
https://www.kaggle.com/datasets/xiaozhouwang/sd2gpt2
https://www.kaggle.com/datasets/xiaozhouwang/sd2hardcode
https://www.kaggle.com/datasets/jeinsong/chatgpt-images-w-prompts <br>
`sd2_meta_laioncoco_p1.csv` - from LAION COCO screening <br>
`sd2_meta_pszemraj.csv` - from https://huggingface.co/datasets/pszemraj/text2image-multi-prompt <br>

## Methods
- Utilized a pre-trained model to generate prompt texts and filtered out prompt sentence embeddings with similarities exceeding 0.8 using vector search via Sentence Transformers
- Constructed image datasets based on generated text prompts using StableDiffusionV2.0
- Employed ConvNext-XXLarge from OpenCLIP, unfreezing 1/4 of layers to extract image embeddings, and evaluated prediction results by cosine similarity to obtain a stable inference model (`train_convnext_xxlarge_littleboat_on3000k_size512.py`)
- Inference: `stablediffusion-littleboat-infer-script.ipynb`
