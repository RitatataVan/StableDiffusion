# Stable Diffusion - Image to Prompts

Goal of the Competition
The goal of this competition is to reverse the typical direction of a generative text-to-image model: instead of generating an image from a text prompt, can you create a model which can predict the text prompt given a generated image? You will make predictions on a dataset containing a wide variety of (prompt, image) pairs generated by Stable Diffusion 2.0, in order to understand how reversible the latent relationship is.
﻿
Your task for this challenge is to predict the prompts that were used to generate target images. Prompts for this challenge were generated using a variety of (non disclosed) methods, and range from fairly simple to fairly complex with multiple objects and modifiers. Images were generated from the prompts using Stable Diffusion 2.0 (768-v-ema.ckpt) and were generated with 50 steps at 768x768 px and then downsized to 512x512 for the competition dataset. (This script was used, with the majority of default parameters unchanged.)
