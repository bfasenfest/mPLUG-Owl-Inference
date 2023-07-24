# mPLUG-Owl-Inference

Batch video inference notebook and script for mPLUG-Owl. Very barebones, but makes it quick to start using the model. Video inference with the default settings takes 15-16 GB VRAM. Lowering the num_frames to 16 or 12 (which may also have better performance) may make it easier for 16GB graphics cards.

# Installtion

Use the `install-mpeg-owl.ipynb` notebook to install mPlug-Owl. If there are issues refer to the [Main Repo](https://github.com/X-PLUG/mPLUG-Owl).

The script replaces the model weights due to the HF model being out of date due to a [NaN bug](https://github.com/X-PLUG/mPLUG-Owl/issues/101)

If you place the videos files you would like processed in the "videos" folder you can run the `mplug-owl-inference.py` script to process all the videos in the folder. Otherwise you can pass arguments to the script to process a different folder.

```
usage: mplug-owl-inference
options:
> -h, --help show this help message and exit
> -f FOLDER, --folder FOLDER
Folder containing videos to caption
> -j JSON, --json JSON
JSON file to save captions to
> -m MAX_LENGTH, --max-length MAX_LENGTH
Max length of captions
> -k TOP_K, --top-k TOP_K
Top k for sampling
> -n NUM_FRAMES, --num-frames NUM_FRAMES
Number of frames to process
> -s SKIP_EXISTING, --skip-existing SKIP_EXISTING
Skip videos that already exist in the JSON
> -D PROMPT, --prompt PROMPT
The main descriptive prompt for the video
> -C PROMPT_CLASSIFICATION, --prompt-classification
The main classifier prompt for the video

```
