from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
from transformers import AutoTokenizer
from mplug_owl_video.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
import torch
import os
import json
import time
import argparse
import nvidia_smi

# Load the model into memory
def load_model():
    start = time.time()
    print('loading model...')
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
    return model, tokenizer, processor

# Process inputs for the model
def process_inputs(model, processor, num_frames, prompts, video_arr):
    inputs = processor(text=prompts, videos=video_arr, num_frames=num_frames, return_tensors='pt')
    inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    return inputs

# Process a video and add it to the JSON
def process_video(video, prompts, label, model, processor, num_frames):
    start = time.time()
    print('processing: ', video)
    video_arr = [video]
    inputs = process_inputs(model, processor, num_frames, prompts, video_arr)
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
            return
    else:
        owl_captions.append({
                'filename': video,
                label: sentence,
        })

    # enable if you want to see the VRAM usage during processing

    # handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    # info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    # print('Model GPU VRAM: ', info.used/1000000000, ' / ', info.total/1000000000)

if __name__ == "__main__":

    #Process Args
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str, default='./videos', help="Folder containing videos to caption")
    parser.add_argument("-j","--json", type=str, default='owl-captions.json', help="JSON file to save captions to")
    parser.add_argument("-m","--max-length", type=int, default=45, help="Max length of captions")
    parser.add_argument("-k","--top-k", type=int, default=5, help="Top k for sampling")
    parser.add_argument("-n","--num-frames", type=int, default=25, help="Number of frames to process")
    parser.add_argument("-s","--skip-existing", type=bool, default=True, help="Skip videos that already exist in the JSON")
    parser.add_argument("-D","--prompt", type=str, default="Describe the video in one sentence", help="The main descriptive prompt for the video")
    parser.add_argument("-C","--prompt-classification", type=str, default='Briefly describe the camera motion and direction in the video, using film production terms', help="The main classifier prompt for the video")
    args = parser.parse_args()

    nvidia_smi.nvmlInit()
    # generate kwargs (the same in transformers) can be passed in the do_generate()
    # adjust max length to change the total length of the response -- lower can be better if you don't want it to get too detailed or hallucinate
    generate_kwargs = {
        'do_sample': True,
        'top_k': args.top_k,
        'max_length': args.max_length,

    }
    # Number of frames to read in the video
    num_frames = args.num_frames

    # Skip files that exist in the json -- if you want to test different values, then change to false so it will overwrite
    skip_existing = args.skip_existing

    # We use a human/AI template to organize the context as a multi-turn conversation.
    # <|video|> denotes an video placehold.
    description_prompts = [
    '''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
    Human: <|video|>
    Human: {}
    AI: '''.format(args.prompt)]
    classifier_prompts = [
    '''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
    Human: <|video|>
    Human: {}
    AI: '''.format(args.prompt_classification)]

    print('max length: ', generate_kwargs['max_length'], '| top_k: ', generate_kwargs['top_k'], '| num_frames: ', num_frames)

    # Load or create JSON to store captions
    json_file = args.json

    if not os.path.exists(json_file):
        with open(json_file, 'w') as f:
            json.dump([], f)
        print('json created')

    with open(json_file, 'r') as f:
        owl_captions = json.load(f)
        print('json read')
    
    # Load the model
    model, tokenizer, processor = load_model()

    if not args.folder.endswith('/'):
        args.folder += '/'
    
    # Process all videos in the videos folder
    video_list = []
    for filename in os.listdir(args.folder):
        #Skip any item already processed in the JSON
        skip = False
        path = args.folder + filename
        for obj in owl_captions:   
            if obj['filename'] == path and skip_existing:
                print(f"Skipping {filename} as it already exists in the JSON")
                skip = True
        if not skip:
            process_video(path, description_prompts, 'description', model, processor, num_frames)
            process_video(path, classifier_prompts, 'shot-type', model, processor, num_frames)

    # Stop logging VRAM usage
    nvidia_smi.nvmlShutdown()

    # Save the JSON
    with open(json_file, 'w') as f:
        json.dump(owl_captions, f)