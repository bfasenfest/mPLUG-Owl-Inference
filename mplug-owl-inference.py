from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
from transformers import AutoTokenizer
from mplug_owl_video.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
import torch
import os
import json
import time
import nvidia_smi

start = time.time()
nvidia_smi.nvmlInit()

#Load the model and tokenizer
pretrained_ckpt = 'model'
model = MplugOwlForConditionalGeneration.from_pretrained(
    pretrained_ckpt,
    torch_dtype=torch.bfloat16,
    device_map={'': 0},
)
image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
tokenizer = AutoTokenizer.from_pretrained(pretrained_ckpt)
processor = MplugOwlProcessor(image_processor, tokenizer)

# Print info about model load time and VRAM usage
end = time.time()
print('Time to load model: ', end-start)

handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
print('Model GPU VRAM: ', info.used/1000000000, ' / ', info.total/1000000000)

# We use a human/AI template to organize the context as a multi-turn conversation.
# <|video|> denotes an video placehold.
description_prompts = [
'''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: <|video|>
Human: Describe the video in one sentence
AI: ''']
classifier_prompts = [
'''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: <|video|>
Human: Briefly describe the camera motion and direction in the video, using film production terms
AI: ''']

# generate kwargs (the same in transformers) can be passed in the do_generate()
# adjust max length to change the total length of the response -- lower can be better if you don't want it to get too detailed or hallucinate
generate_kwargs = {
    'do_sample': True,
    'top_k': 5,
    'max_length': 45

}
# Number of frames to read in the video
num_frames=25

# Skip files that exist in the json -- if you want to test different values, then change to false so it will overwrite
skip_existing = False

print('max length: ', generate_kwargs['max_length'], '| top_k: ', generate_kwargs['top_k'], '| num_frames: ', num_frames)

# Load or create JSON to store captions
json_file = 'owl-captions.json'

if not os.path.exists(json_file):
    with open(json_file, 'w') as f:
        json.dump([], f)
    print('json created')

with open(json_file, 'r') as f:
    owl_captions = json.load(f)
    print('json read')

# Process inputs for the model
def process_inputs(prompts, video_arr):
    inputs = processor(text=prompts, videos=video_arr, num_frames=num_frames, return_tensors='pt')
    inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    return inputs

# Process a video and add it to the JSON
def process_video(video, prompts, label):
    start = time.time()
    print('processing: ', video)
    video_arr = [video]
    inputs = process_inputs(prompts, video_arr)
    with torch.no_grad():
        res = model.generate(**inputs, **generate_kwargs)
    sentence = tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
    print(label, ": ", sentence)
    
    end = time.time()
    print('Time to process video: ', end-start)
    # Check if an object with the same filename already exists in owl_captions
    for obj in owl_captions:
        if obj['filename'] == video:
            obj[label] = sentence
            print('append to existing')
            return
    else:
        print ('create new object')        
        owl_captions.append({
                'filename': video,
                label: sentence,
        })

    # enable if you want to see the VRAM usage during processing

    # handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    # info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    # print('Model GPU VRAM: ', info.used/1000000000, ' / ', info.total/1000000000)

# Process all videos in the videos folder
video_list = []
folder_path = './videos/'
for filename in os.listdir(folder_path):
    #Skip any item already processed in the JSON
    skip = False
    path = folder_path + filename
    for obj in owl_captions:   
        if obj['filename'] == path and skip_existing:
            print(f"Skipping {filename} as it already exists in the JSON")
            skip = True
    if not skip:
        process_video(path, description_prompts, 'description')
        process_video(path, classifier_prompts, 'shot-type')

# Stop logging VRAM usage
nvidia_smi.nvmlShutdown()

# Save the JSON
with open(json_file, 'w') as f:
    json.dump(owl_captions, f)



