from collections.abc import Callable
import torch
import torchvision.transforms.functional as F
import io
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageColor, ImageFont
import random
import numpy as np
import re
from pathlib import Path

from safetensors.torch import save_file
import comfy.ops
import comfy.model_patcher

def apply_florence2_lora(patcher, lora_path, strength=1.0):
    """Apply a peft-style LoRA adapter to a Florence2 ModelPatcher."""
    import json
    lora_sd = load_torch_file(os.path.join(lora_path, "adapter_model.safetensors"))
    config = json.load(open(os.path.join(lora_path, "adapter_config.json")))
    lora_alpha = config.get("lora_alpha", 32)

    # Convert peft keys (base_model.model.X.lora_A/B.weight) to comfy lora format (X.lora_down/up.weight)
    converted = {}
    for k, v in lora_sd.items():
        k = k.replace("base_model.model.", "")
        k = k.replace("lora_A", "lora_down").replace("lora_B", "lora_up")
        converted[k] = v

    # Add alpha entries so comfy's lora loader applies correct scaling
    seen_prefixes = set()
    for k in list(converted.keys()):
        prefix = k.rsplit(".", 2)[0]  # e.g. "language_model.model.encoder.layers.0.self_attn.q_proj"
        if prefix not in seen_prefixes:
            seen_prefixes.add(prefix)
            converted[f"{prefix}.alpha"] = torch.tensor(lora_alpha, dtype=torch.float32)

    # Build key map: lora prefix -> model weight key
    model_sd = patcher.model.state_dict()
    key_map = {}
    for k in model_sd:
        lora_prefix = k.replace(".weight", "")
        if f"{lora_prefix}.lora_down.weight" in converted:
            key_map[lora_prefix] = k

    import comfy.lora
    patches = comfy.lora.load_lora(converted, key_map)
    patcher.add_patches(patches, strength_patch=strength)
    return patcher


def load_florence2(model_path, dtype):
    from .model.config import Florence2Config
    from .model.model import Florence2
    from .model.processing import Processor

    load_device = mm.text_encoder_device()
    offload_device = mm.text_encoder_offload_device()

    config_path = os.path.join(model_path, "config.json")
    checkpoint_path = os.path.join(model_path, "model.safetensors")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(model_path, "pytorch_model.bin")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No model weights found at {model_path}")

    if os.path.exists(config_path):
        config = Florence2Config.from_json(config_path)
    else:
        sd = load_torch_file(checkpoint_path)
        config = Florence2Config.from_state_dict(sd)
        sd = None

    model = Florence2(config, dtype=dtype, device="cpu", operations=comfy.ops.manual_cast)

    sd = load_torch_file(checkpoint_path)
    for key in ["language_model.model.encoder.embed_tokens.weight",
                "language_model.model.decoder.embed_tokens.weight"]:
        if key in sd and "language_model.model.shared.weight" in sd:
            sd.pop(key, None)

    m, u = model.load_state_dict(sd, strict=False)
    if m:
        print(f"Florence2 missing keys: {m}")
    if u:
        print(f"Florence2 unexpected keys: {u}")
    del sd

    model.language_model.tie_weights()
    model = model.eval()

    patcher = comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)

    processor = Processor(model_path=model_path)
    return patcher, processor


def create_path_dict(paths: list[str], predicate: Callable[[Path], bool] = lambda _: True) -> dict[str, str]:
    """
    Creates a flat dictionary of the contents of all given paths: ``{name: absolute_path}``.

    Non-recursive.  Optionally takes a predicate to filter items.  Duplicate names overwrite (the last one wins).

    Args:
        paths (list[str]):
            The paths to search for items.
        predicate (Callable[[Path], bool]): 
            (Optional) If provided, each path is tested against this filter.
            Returns ``True`` to include a path.

            Default: Include everything
    """

    flattened_paths = [item for path in paths if Path(path).exists() for item in Path(path).iterdir() if predicate(item)]

    return {item.name: str(item.absolute()) for item in flattened_paths}


import comfy.model_management as mm
from comfy.utils import ProgressBar, load_torch_file

device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

import folder_paths

script_directory = os.path.dirname(os.path.abspath(__file__))
model_directory = os.path.join(folder_paths.models_dir, "LLM")
os.makedirs(model_directory, exist_ok=True)

# Ensure ComfyUI knows about the LLM model path
folder_paths.add_model_folder_path("LLM", model_directory)


model_list = [
            'microsoft/Florence-2-base',
            'microsoft/Florence-2-base-ft',
            'microsoft/Florence-2-large',
            'microsoft/Florence-2-large-ft',
            'HuggingFaceM4/Florence-2-DocVQA',
            'thwri/CogFlorence-2.1-Large',
            'thwri/CogFlorence-2.2-Large',
            'gokaygokay/Florence-2-SD3-Captioner',
            'gokaygokay/Florence-2-Flux-Large',
            'MiaoshouAI/Florence-2-base-PromptGen-v1.5',
            'MiaoshouAI/Florence-2-large-PromptGen-v1.5',
            'MiaoshouAI/Florence-2-base-PromptGen-v2.0',
            'MiaoshouAI/Florence-2-large-PromptGen-v2.0',
            'PJMixers-Images/Florence-2-base-Castollux-v0.5'
            ]

class DownloadAndLoadFlorence2Model:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (model_list, {"default": 'microsoft/Florence-2-base'}),
            "precision": ([ 'fp16','bf16','fp32'],
                    {
                    "default": 'fp16'
                    }),
            },
            "optional": {
                "lora": ("PEFTLORA",),
                "convert_to_safetensors": ("BOOLEAN", {"default": False, "tooltip": "Some of the older model weights are not saved in .safetensors format, which seem to cause longer loading times, this option converts the .bin weights to .safetensors"}),
            }
        }

    RETURN_TYPES = ("FL2MODEL",)
    RETURN_NAMES = ("florence2_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "Florence2"

    def loadmodel(self, model, precision, lora=None, convert_to_safetensors=False, attention=None, **kwargs):
        if model not in model_list:
            raise ValueError(f"Model {model} is not in the supported model list.")

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        model_name = model.rsplit('/', 1)[-1]
        model_path = os.path.join(model_directory, model_name)

        if not os.path.exists(model_path):
            print(f"Downloading Florence2 model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model,
                            local_dir=model_path,
                            local_dir_use_symlinks=False)

        if convert_to_safetensors:
            model_weight_path = os.path.join(model_path, 'pytorch_model.bin')
            if os.path.exists(model_weight_path):
                safetensors_weight_path = os.path.join(model_path, 'model.safetensors')
                print(f"Converting {model_weight_path} to {safetensors_weight_path}")
                if not os.path.exists(safetensors_weight_path):
                    sd = torch.load(model_weight_path, map_location=offload_device)
                    sd_new = {}
                    for k, v in sd.items():
                        sd_new[k] = v.clone()
                    save_file(sd_new, safetensors_weight_path)
                    if os.path.exists(safetensors_weight_path):
                        print(f"Conversion successful. Deleting original file: {model_weight_path}")
                        os.remove(model_weight_path)
                        print(f"Original {model_weight_path} file deleted.")

        patcher, processor = load_florence2(model_path, dtype)

        if lora is not None:
            apply_florence2_lora(patcher, lora)

        florence2_model = {
            'patcher': patcher,
            'processor': processor,
            'dtype': dtype
            }

        return (florence2_model,)

class DownloadAndLoadFlorence2Lora:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (
                    [ 
                    'NikshepShetty/Florence-2-pixelprose',
                    ],
                  ),            
            },
          
        }

    RETURN_TYPES = ("PEFTLORA",)
    RETURN_NAMES = ("lora",)
    FUNCTION = "loadmodel"
    CATEGORY = "Florence2"

    def loadmodel(self, model):
        if model not in ['NikshepShetty/Florence-2-pixelprose']:
            raise ValueError(f"Lora Model {model} is not in the supported lora model list.")
        model_name = model.rsplit('/', 1)[-1]
        model_path = os.path.join(model_directory, model_name)
        
        if not os.path.exists(model_path):
            print(f"Downloading Florence2 lora model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model,
                            local_dir=model_path,
                            local_dir_use_symlinks=False)
        return (model_path,)
    
class Florence2ModelLoader:

    @classmethod
    def INPUT_TYPES(s):
        all_llm_paths = folder_paths.get_folder_paths("LLM")
        s.model_paths = create_path_dict(all_llm_paths, lambda x: x.is_dir())

        return {"required": {
            "model": ([*s.model_paths], {"tooltip": "models are expected to be in Comfyui/models/LLM folder"}),
            "precision": (['fp16','bf16','fp32'],),
            },
            "optional": {
                "lora": ("PEFTLORA",),
                "convert_to_safetensors": ("BOOLEAN", {"default": False, "tooltip": "Some of the older model weights are not saved in .safetensors format, which seem to cause longer loading times, this option converts the .bin weights to .safetensors"}),
            }
        }

    RETURN_TYPES = ("FL2MODEL",)
    RETURN_NAMES = ("florence2_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "Florence2"

    def loadmodel(self, model, precision, lora=None, convert_to_safetensors=False, attention=None, **kwargs):
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        model_path = Florence2ModelLoader.model_paths.get(model)
        print(f"Loading model from {model_path}")
        if convert_to_safetensors:
            model_weight_path = os.path.join(model_path, 'pytorch_model.bin')
            if os.path.exists(model_weight_path):
                safetensors_weight_path = os.path.join(model_path, 'model.safetensors')
                print(f"Converting {model_weight_path} to {safetensors_weight_path}")
                if not os.path.exists(safetensors_weight_path):
                    sd = torch.load(model_weight_path, map_location=offload_device)
                    sd_new = {}
                    for k, v in sd.items():
                        sd_new[k] = v.clone()
                    save_file(sd_new, safetensors_weight_path)
                    if os.path.exists(safetensors_weight_path):
                        print(f"Conversion successful. Deleting original file: {model_weight_path}")
                        os.remove(model_weight_path)
                        print(f"Original {model_weight_path} file deleted.")

        patcher, processor = load_florence2(model_path, dtype)

        if lora is not None:
            apply_florence2_lora(patcher, lora)

        florence2_model = {
            'patcher': patcher,
            'processor': processor,
            'dtype': dtype
            }

        return (florence2_model,)

class Florence2Run:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "florence2_model": ("FL2MODEL", ),
                "text_input": ("STRING", {"default": "", "multiline": True}),
                "task": (
                    [ 
                    'region_caption',
                    'dense_region_caption',
                    'region_proposal',
                    'caption',
                    'detailed_caption',
                    'more_detailed_caption',
                    'caption_to_phrase_grounding',
                    'referring_expression_segmentation',
                    'ocr',
                    'ocr_with_region',
                    'docvqa',
                    'prompt_gen_tags',
                    'prompt_gen_mixed_caption',
                    'prompt_gen_analyze',
                    'prompt_gen_mixed_caption_plus',
                    ],
                   ),
                "fill_mask": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "max_new_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
                "num_beams": ("INT", {"default": 3, "min": 1, "max": 64}),
                "do_sample": ("BOOLEAN", {"default": True}),
                "output_mask_select": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "JSON")
    RETURN_NAMES =("image", "mask", "caption", "data") 
    FUNCTION = "encode"
    CATEGORY = "Florence2"

    def hash_seed(self, seed):
        import hashlib
        # Convert the seed to a string and then to bytes
        seed_bytes = str(seed).encode('utf-8')
        # Create a SHA-256 hash of the seed bytes
        hash_object = hashlib.sha256(seed_bytes)
        # Convert the hash to an integer
        hashed_seed = int(hash_object.hexdigest(), 16)
        # Ensure the hashed seed is within the acceptable range for set_seed
        return hashed_seed % (2**32)

    def encode(self, image, text_input, florence2_model, task, fill_mask, keep_model_loaded=False, 
            num_beams=3, max_new_tokens=1024, do_sample=True, output_mask_select="", seed=None):
        _, height, width, _ = image.shape
        annotated_image_tensor = None
        mask_tensor = None
        processor = florence2_model['processor']
        patcher = florence2_model['patcher']
        dtype = florence2_model['dtype']
        mm.load_model_gpu(patcher)
        model = patcher.model

        if seed:
            torch.manual_seed(self.hash_seed(seed))

        colormap = ['blue','orange','green','purple','brown','pink','olive','cyan','red',
                    'lime','indigo','violet','aqua','magenta','gold','tan','skyblue']

        prompts = {
            'region_caption': '<OD>',
            'dense_region_caption': '<DENSE_REGION_CAPTION>',
            'region_proposal': '<REGION_PROPOSAL>',
            'caption': '<CAPTION>',
            'detailed_caption': '<DETAILED_CAPTION>',
            'more_detailed_caption': '<MORE_DETAILED_CAPTION>',
            'caption_to_phrase_grounding': '<CAPTION_TO_PHRASE_GROUNDING>',
            'referring_expression_segmentation': '<REFERRING_EXPRESSION_SEGMENTATION>',
            'ocr': '<OCR>',
            'ocr_with_region': '<OCR_WITH_REGION>',
            'docvqa': '<DocVQA>',
            'prompt_gen_tags': '<GENERATE_TAGS>',
            'prompt_gen_mixed_caption': '<MIXED_CAPTION>',
            'prompt_gen_analyze': '<ANALYZE>',
            'prompt_gen_mixed_caption_plus': '<MIXED_CAPTION_PLUS>',
        }
        task_prompt = prompts.get(task, '<OD>')

        if (task not in ['referring_expression_segmentation', 'caption_to_phrase_grounding', 'docvqa']) and text_input:
            raise ValueError("Text input (prompt) is only supported for 'referring_expression_segmentation', 'caption_to_phrase_grounding', and 'docvqa'")

        if text_input != "":
            prompt = task_prompt + " " + text_input
        else:
            prompt = task_prompt

        image = image.permute(0, 3, 1, 2)

        out = []
        out_masks = []
        out_results = []
        out_data = []
        pbar = ProgressBar(len(image))
        for img in image:
            image_pil = F.to_pil_image(img)
            inputs = processor(text=prompt, images=img.unsqueeze(0))
            load_device = patcher.load_device

            generated_ids = model.generate(
                input_ids=inputs["input_ids"].to(load_device),
                pixel_values=inputs["pixel_values"].to(dtype=dtype, device=load_device),
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                num_beams=num_beams,
            )

            results = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            print(results)
            # cleanup the special tokens from the final list
            if task == 'ocr_with_region':
                clean_results = str(results)       
                cleaned_string = re.sub(r'</?s>|<[^>]*>', '\n',  clean_results)
                clean_results = re.sub(r'\n+', '\n', cleaned_string)
            else:
                clean_results = str(results)       
                clean_results = clean_results.replace('</s>', '')
                clean_results = clean_results.replace('<s>', '')

             #return single string if only one image for compatibility with nodes that can't handle string lists
            if len(image) == 1:
                out_results = clean_results
            else:
                out_results.append(clean_results)

            W, H = image_pil.size
            
            parsed_answer = processor.post_process_generation(results, task=task_prompt, image_size=(W, H))

            if task == 'region_caption' or task == 'dense_region_caption' or task == 'caption_to_phrase_grounding' or task == 'region_proposal':           
                fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
                fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
                ax.imshow(image_pil)
                bboxes = parsed_answer[task_prompt]['bboxes']
                labels = parsed_answer[task_prompt]['labels']

                mask_indexes = []
                # Determine mask indexes outside the loop
                if output_mask_select != "":
                    mask_indexes = [n for n in output_mask_select.split(",")]
                    print(mask_indexes)
                else:
                    mask_indexes = [str(i) for i in range(len(bboxes))]

                # Initialize mask_layer only if needed
                if fill_mask:
                    mask_layer = Image.new('RGB', image_pil.size, (0, 0, 0))
                    mask_draw = ImageDraw.Draw(mask_layer)

                for index, (bbox, label) in enumerate(zip(bboxes, labels)):
                    # Modify the label to include the index
                    indexed_label = f"{index}.{label}"
                    
                    if fill_mask:
                        # Ensure y1 is greater than or equal to y0 for mask drawing
                        x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
                        if y1 < y0:
                            y0, y1 = y1, y0
                        if x1 < x0:
                            x0, x1 = x1, x0
                            
                        if str(index) in mask_indexes:
                            print("match index:", str(index), "in mask_indexes:", mask_indexes)
                            mask_draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255))
                        if label in mask_indexes:
                            print("match label")
                            mask_draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255))

                    # Create a Rectangle patch
                    # Ensure y1 is greater than or equal to y0
                    y0, y1 = bbox[1], bbox[3]
                    if y1 < y0:
                        y0, y1 = y1, y0
                    
                    rect = patches.Rectangle(
                        (bbox[0], y0),  # (x,y) - lower left corner
                        bbox[2] - bbox[0],   # Width
                        y1 - y0,   # Height
                        linewidth=1,
                        edgecolor='r',
                        facecolor='none',
                        label=indexed_label
                    )
                     # Calculate text width with a rough estimation
                    text_width = len(label) * 6  # Adjust multiplier based on your font size
                    text_height = 12  # Adjust based on your font size

                    # Get corrected coordinates
                    x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
                    if y1 < y0:
                        y0, y1 = y1, y0
                    if x1 < x0:
                        x0, x1 = x1, x0

                    # Initial text position
                    text_x = x0
                    text_y = y0 - text_height  # Position text above the top-left of the bbox

                    # Adjust text_x if text is going off the left or right edge
                    if text_x < 0:
                        text_x = 0
                    elif text_x + text_width > W:
                        text_x = W - text_width

                    # Adjust text_y if text is going off the top edge
                    if text_y < 0:
                        text_y = y1  # Move text below the bottom-left of the bbox if it doesn't overlap with bbox

                    # Add the rectangle to the plot
                    ax.add_patch(rect)
                    facecolor = random.choice(colormap) if len(image) == 1 else 'red'
                    # Add the label
                    plt.text(
                        text_x,
                        text_y,
                        indexed_label,
                        color='white',
                        fontsize=12,
                        bbox=dict(facecolor=facecolor, alpha=0.5)
                    )
                if fill_mask:             
                    mask_tensor = F.to_tensor(mask_layer)
                    mask_tensor = mask_tensor.unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
                    mask_tensor = mask_tensor.mean(dim=0, keepdim=True)
                    mask_tensor = mask_tensor.repeat(1, 1, 1, 3)
                    mask_tensor = mask_tensor[:, :, :, 0]
                    out_masks.append(mask_tensor)           

                # Remove axis and padding around the image
                ax.axis('off')
                ax.margins(0,0)
                ax.get_xaxis().set_major_locator(plt.NullLocator())
                ax.get_yaxis().set_major_locator(plt.NullLocator())
                fig.canvas.draw() 
                buf = io.BytesIO()
                plt.savefig(buf, format='png', pad_inches=0)
                buf.seek(0)
                annotated_image_pil = Image.open(buf)

                annotated_image_tensor = F.to_tensor(annotated_image_pil)
                out_tensor = annotated_image_tensor[:3, :, :].unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
                out.append(out_tensor)
               
                if task == 'caption_to_phrase_grounding':
                    out_data.append(parsed_answer[task_prompt])
                else:
                    out_data.append(bboxes)

                
                pbar.update(1)
    
                plt.close(fig)

            elif task == 'referring_expression_segmentation':
                # Create a new black image
                mask_image = Image.new('RGB', (W, H), 'black')
                mask_draw = ImageDraw.Draw(mask_image)
  
                predictions = parsed_answer[task_prompt]
    
                # Iterate over polygons and labels  
                for polygons, label in zip(predictions['polygons'], predictions['labels']):
                    color = random.choice(colormap)
                    for _polygon in polygons:  
                        _polygon = np.array(_polygon).reshape(-1, 2)
                        # Clamp polygon points to image boundaries
                        _polygon = np.clip(_polygon, [0, 0], [W - 1, H - 1])
                        if len(_polygon) < 3:  
                            print('Invalid polygon:', _polygon)
                            continue  
                        
                        _polygon = _polygon.reshape(-1).tolist()
                        
                        # Draw the polygon
                        if fill_mask:
                            overlay = Image.new('RGBA', image_pil.size, (255, 255, 255, 0))
                            image_pil = image_pil.convert('RGBA')
                            draw = ImageDraw.Draw(overlay)
                            color_with_opacity = ImageColor.getrgb(color) + (180,)
                            draw.polygon(_polygon, outline=color, fill=color_with_opacity, width=3)
                            image_pil = Image.alpha_composite(image_pil, overlay)
                        else:
                            draw = ImageDraw.Draw(image_pil)
                            draw.polygon(_polygon, outline=color, width=3)

                        #draw mask
                        mask_draw.polygon(_polygon, outline="white", fill="white")
                        
                image_tensor = F.to_tensor(image_pil)
                image_tensor = image_tensor[:3, :, :].unsqueeze(0).permute(0, 2, 3, 1).cpu().float() 
                out.append(image_tensor)

                mask_tensor = F.to_tensor(mask_image)
                mask_tensor = mask_tensor.unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
                mask_tensor = mask_tensor.mean(dim=0, keepdim=True)
                mask_tensor = mask_tensor.repeat(1, 1, 1, 3)
                mask_tensor = mask_tensor[:, :, :, 0]
                out_masks.append(mask_tensor)
                pbar.update(1)

            elif task == 'ocr_with_region':
                try:
                    font = ImageFont.load_default().font_variant(size=24)
                except:
                    font = ImageFont.load_default()
                predictions = parsed_answer[task_prompt]
                scale = 1
                image_pil = image_pil.convert('RGBA')
                overlay = Image.new('RGBA', image_pil.size, (255, 255, 255, 0))
                draw = ImageDraw.Draw(overlay)
                bboxes, labels = predictions['quad_boxes'], predictions['labels']
                
                # Create a new black image for the mask
                mask_image = Image.new('RGB', (W, H), 'black')
                mask_draw = ImageDraw.Draw(mask_image)
                
                for box, label in zip(bboxes, labels):
                    scaled_box = [v / (width if idx % 2 == 0 else height) for idx, v in enumerate(box)]
                    out_data.append({"label": label, "box": scaled_box})
                    
                    color = random.choice(colormap)
                    new_box = (np.array(box) * scale).tolist()
                    
                    # Ensure polygon coordinates are valid
                    # For polygons, we need to make sure the points form a valid shape
                    # This is a simple check to ensure the polygon has at least 3 points
                    if len(new_box) >= 6:  # At least 3 points (x,y pairs)
                        if fill_mask:
                            color_with_opacity = ImageColor.getrgb(color) + (180,)
                            draw.polygon(new_box, outline=color, fill=color_with_opacity, width=3)
                        else:
                            draw.polygon(new_box, outline=color, width=3)
                        
                        # Get the first point for text positioning
                        text_x, text_y = new_box[0]+8, new_box[1]+2
                        
                        draw.text((text_x, text_y),
                                  "{}".format(label),
                                  align="right",
                                  font=font,
                                  fill=color)
                        
                        # Draw the mask
                        mask_draw.polygon(new_box, outline="white", fill="white")
                
                image_pil = Image.alpha_composite(image_pil, overlay)
                image_pil = image_pil.convert('RGB')
                
                image_tensor = F.to_tensor(image_pil)
                image_tensor = image_tensor[:3, :, :].unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
                out.append(image_tensor)

                # Process the mask
                mask_tensor = F.to_tensor(mask_image)
                mask_tensor = mask_tensor.unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
                mask_tensor = mask_tensor.mean(dim=0, keepdim=True)
                mask_tensor = mask_tensor.repeat(1, 1, 1, 3)
                mask_tensor = mask_tensor[:, :, :, 0]
                out_masks.append(mask_tensor)

                pbar.update(1)
            
            elif task == 'docvqa':
                if text_input == "":
                    raise ValueError("Text input (prompt) is required for 'docvqa'")
                prompt = "<DocVQA> " + text_input

                inputs = processor(text=prompt, images=img.unsqueeze(0))
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"].to(load_device),
                    pixel_values=inputs["pixel_values"].to(dtype=dtype, device=load_device),
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    num_beams=num_beams,
                )

                results = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                clean_results = results.replace('</s>', '').replace('<s>', '')
                
                if len(image) == 1:
                    out_results = clean_results
                else:
                    out_results.append(clean_results)
                    
                out.append(F.to_tensor(image_pil).unsqueeze(0).permute(0, 2, 3, 1).cpu().float())

                pbar.update(1)
            
        if len(out) > 0:
            out_tensor = torch.cat(out, dim=0)
        else:
            out_tensor = torch.zeros((1, 64,64, 3), dtype=torch.float32, device="cpu")
        if len(out_masks) > 0:
            out_mask_tensor = torch.cat(out_masks, dim=0)
        else:
            out_mask_tensor = torch.zeros((1,64,64), dtype=torch.float32, device="cpu")

        if not keep_model_loaded:
            keep = [x for x in mm.current_loaded_models if x.model is not patcher]
            mm.free_memory(1e30, patcher.load_device, keep_loaded=keep)
            mm.soft_empty_cache()
        
        return (out_tensor, out_mask_tensor, out_results, out_data)
     
NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadFlorence2Model": DownloadAndLoadFlorence2Model,
    "DownloadAndLoadFlorence2Lora": DownloadAndLoadFlorence2Lora,
    "Florence2ModelLoader": Florence2ModelLoader,
    "Florence2Run": Florence2Run,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadFlorence2Model": "DownloadAndLoadFlorence2Model",
    "DownloadAndLoadFlorence2Lora": "DownloadAndLoadFlorence2Lora",
    "Florence2ModelLoader": "Florence2ModelLoader",
    "Florence2Run": "Florence2Run",
}
