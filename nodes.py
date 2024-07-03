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


#workaround for unnecessary flash_attn requirement
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports


import comfy.model_management as mm
from comfy.utils import ProgressBar
import folder_paths

script_directory = os.path.dirname(os.path.abspath(__file__))

from transformers import AutoModelForCausalLM, AutoProcessor

class DownloadAndLoadFlorence2Model:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (
                    [ 
                    'microsoft/Florence-2-base',
                    'microsoft/Florence-2-base-ft',
                    'microsoft/Florence-2-large',
                    'microsoft/Florence-2-large-ft',
                    'HuggingFaceM4/Florence-2-DocVQA'
                    ],
                    {
                    "default": 'microsoft/Florence-2-base'
                    }),
            "precision": ([ 'fp16','bf16','fp32'],
                    {
                    "default": 'fp16'
                    }),
            "attention": (
                    [ 'flash_attention_2', 'sdpa', 'eager'],
                    {
                    "default": 'sdpa'
                    }),

            },
        }

    RETURN_TYPES = ("FL2MODEL",)
    RETURN_NAMES = ("florence2_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "Florence2"

    def loadmodel(self, model, precision, attention):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        model_name = model.rsplit('/', 1)[-1]
        model_path = os.path.join(folder_paths.models_dir, "LLM", model_name)
        
        if not os.path.exists(model_path):
            print(f"Downloading Lumina model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model,
                            local_dir=model_path,
                            local_dir_use_symlinks=False)
            
        print(f"using {attention} for attention")
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports): #workaround for unnecessary flash_attn requirement
            model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation=attention, device_map=device, torch_dtype=dtype,trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        florence2_model = {
            'model': model, 
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
                    'docvqa'
                    ],
                   ),
                "fill_mask": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "max_new_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
                "num_beams": ("INT", {"default": 3, "min": 1, "max": 64}),
                "do_sample": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "STRING",)
    RETURN_NAMES =("image", "mask", "caption",)
    OUTPUT_IS_LIST = (False, False, True,)
    FUNCTION = "encode"
    CATEGORY = "Florence2"

    def encode(self, image, text_input, florence2_model, task, fill_mask, keep_model_loaded=False, 
            num_beams=3, max_new_tokens=1024, do_sample=True):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        annotated_image_tensor = None
        mask_tensor = None
        processor = florence2_model['processor']
        model = florence2_model['model']
        dtype = florence2_model['dtype']
        model.to(device)

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
            'docvqa': '<DocVQA>'
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
        out_results: list[str] = []
        pbar = ProgressBar(len(image))
        for img in image:
            image_pil = F.to_pil_image(img)
            inputs = processor(text=prompt, images=image_pil, return_tensors="pt", do_rescale=False).to(dtype).to(device)

            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
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
                out_results = [clean_results]
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

                if fill_mask:
                    mask_layer = Image.new('RGB', image_pil.size, (0, 0, 0))
                    mask_draw = ImageDraw.Draw(mask_layer)

                # Loop through the bounding boxes and labels and add them to the plot
                for bbox, label in zip(bboxes, labels):

                    if fill_mask:
                        # Draw a mask on the bbox area
                        mask_color = (255, 255, 255)  # White with half opacity
                        mask_draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], fill=mask_color)
                    # Create a Rectangle patch
                    rect = patches.Rectangle(
                        (bbox[0], bbox[1]),  # (x,y) - lower left corner
                        bbox[2] - bbox[0],   # Width
                        bbox[3] - bbox[1],   # Height
                        linewidth=1,
                        edgecolor='r',
                        facecolor='none',
                        label=label
                    )
                     # Calculate text width with a rough estimation
                    text_width = len(label) * 6  # Adjust multiplier based on your font size
                    text_height = 12  # Adjust based on your font size

                    # Initial text position
                    text_x = bbox[0]
                    text_y = bbox[1] - text_height  # Position text above the top-left of the bbox

                    # Adjust text_x if text is going off the left or right edge
                    if text_x < 0:
                        text_x = 0
                    elif text_x + text_width > W:
                        text_x = W - text_width

                    # Adjust text_y if text is going off the top edge
                    if text_y < 0:
                        text_y = bbox[3]  # Move text below the bottom-left of the bbox if it doesn't overlap with bbox

                    # Add the rectangle to the plot
                    ax.add_patch(rect)
                    facecolor = random.choice(colormap) if len(image) == 1 else 'red'
                    # Add the label
                    plt.text(
                        text_x,
                        text_y,
                        label,
                        color='white',
                        fontsize=12,
                        bbox=dict(facecolor=facecolor, alpha=0.5)
                    )
                if fill_mask:
                    # Combine the original image with the mask layer
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
                draw = ImageDraw.Draw(image_pil)
                bboxes, labels = predictions['quad_boxes'], predictions['labels']
                for box, label in zip(bboxes, labels):
                    color = random.choice(colormap)
                    new_box = (np.array(box) * scale).tolist()
                    draw.polygon(new_box, width=3, outline=color)
                    draw.text((new_box[0]+8, new_box[1]+2),
                                "{}".format(label),
                                align="right",
                                font=font,
                                fill=color)
                image_tensor = F.to_tensor(image_pil)
                image_tensor = image_tensor[:3, :, :].unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
                out.append(image_tensor)

            elif task == 'docvqa':
                if text_input == "":
                    raise ValueError("Text input (prompt) is required for 'docvqa'")
                prompt = "<DocVQA> " + text_input

                inputs = processor(text=prompt, images=image_pil, return_tensors="pt", do_rescale=False).to(dtype).to(device)
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    num_beams=num_beams,
                )

                results = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                clean_results = results.replace('</s>', '').replace('<s>', '')
                
                if len(image) == 1:
                    out_results = [clean_results]
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
            print("Offloading model...")
            model.to(offload_device)
            mm.soft_empty_cache()
        
        return (out_tensor, out_mask_tensor, out_results,)
     
NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadFlorence2Model": DownloadAndLoadFlorence2Model,
    "Florence2Run": Florence2Run,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadFlorence2Model": "DownloadAndLoadFlorence2Model",
    "Florence2Run": "Florence2Run",
}
