import os
import nodes
from aiohttp import web
from server import PromptServer


def get_comfy_dir(subpath=None):
    import inspect
    dir = os.path.dirname(inspect.getfile(PromptServer))
    if subpath is not None:
        dir = os.path.join(dir, subpath)

    dir = os.path.abspath(dir)

    return dir


@PromptServer.instance.routes.get("/florence2/tag")
async def get_tags(request):
    if "filename" not in request.rel_url.query:
        return web.Response(status=404)

    type = request.query.get("type", "input")
    if type not in ["output", "input", "temp"]:
        return web.Response(status=400)

    target_dir = get_comfy_dir(type)
    image_path = os.path.abspath(
        os.path.join(
            target_dir,
            request.query.get("subfolder", ""),
            request.query["filename"])
    )
    if os.path.commonpath((image_path, target_dir)) != target_dir:
        return web.Response(status=403)

    if not os.path.isfile(image_path):
        return web.Response(status=404)

    image, _ = nodes.LoadImage().load_image(
        os.path.join(request.query.get("subfolder", ""), request.query["filename"])
        + f" [{type}]")
    model = request.query.get("model", "microsoft/Florence-2-base")
    task = request.query.get("task", "caption")
    text_input = request.query.get("text_input", "")
    output_mask_select = request.query.get("output_mask_select", "")
    fill_mask = request.query.get("fill_mask", True)
    do_sample = request.query.get("do_sample", True)
    keep_model_loaded = request.query.get("keep_model_loaded", True)
    num_beams = request.query.get("num_beams", 1)
    max_new_tokens = request.query.get("max_new_tokens", 1024)
    florence2_model, = nodes.DownloadAndLoadFlorence2Model().loadmodel(model=model, precision="fp16", attention="sdpa")
    _, _, res, _ = nodes.Florence2Run().encode(
        image=image,
        text_input=text_input,
        florence2_model=florence2_model,
        task=task,
        output_mask_select=output_mask_select,
        do_sample=do_sample,
        fill_mask=fill_mask,
        keep_model_loaded=keep_model_loaded,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams
    )
    return web.json_response(res)
