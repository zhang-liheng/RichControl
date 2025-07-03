from os import makedirs, path
from time import time
from argparse import ArgumentParser
from copy import deepcopy
import yaml
from omegaconf import OmegaConf
import torch
import openai
from diffusers import DDIMScheduler, StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

from pipelines.pipeline_sdxl import RichControlStableDiffusionXLPipeline
from utils import *


@torch.no_grad()
def inference(
    pipe, refiner, device,
    structure_image, appearance_image,
    prompt, structure_prompt, appearance_prompt,
    positive_prompt, negative_prompt,
    guidance_scale, structure_guidance_scale, appearance_guidance_scale,
    num_inference_steps, eta, seed,
    width, height,
    structure_schedule, appearance_schedule,
    model_config,
):  
    seed_everything(seed)
    
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    
    control_config = get_control_config(structure_schedule, appearance_schedule)
    
    config = yaml.safe_load(control_config)
    register_control(
        model = pipe,
        timesteps = timesteps,
        control_schedule = config["control_schedule"],
        control_target = config["control_target"],
        )
    
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    
    self_recurrence_schedule = get_self_recurrence_schedule(config["self_recurrence_schedule"], num_inference_steps)
    
    pipe.set_progress_bar_config(desc="RichControl inference")
    result, structure, appearance = pipe(
        prompt = prompt,
        structure_prompt = structure_prompt,
        appearance_prompt = appearance_prompt,
        structure_image = structure_image,
        appearance_image = appearance_image,
        num_inference_steps = num_inference_steps,
        negative_prompt = negative_prompt,
        positive_prompt = positive_prompt,
        height = height,
        width = width,
        guidance_scale = guidance_scale,
        structure_guidance_scale = structure_guidance_scale,
        appearance_guidance_scale = appearance_guidance_scale,
        eta = eta,
        output_type = "pil",
        return_dict = False,
        control_schedule = config["control_schedule"],
        self_recurrence_schedule = self_recurrence_schedule,
        model_config = model_config,
    )
    
    if refiner is not None:
        refiner.set_progress_bar_config(desc="Refiner")
        result_refiner = refiner(
            image = pipe.refiner_args["latents"],
            prompt = pipe.refiner_args["prompt"],
            negative_prompt = pipe.refiner_args["negative_prompt"],
            height = height,
            width = width,
            num_inference_steps = num_inference_steps,
            guidance_scale = guidance_scale,
            guidance_rescale = 0.7,
            num_images_per_prompt = 1,
            eta = eta,
            output_type = "pil",
        ).images
    
    else:
        result_refiner = [None]

    del pipe.refiner_args

    return result[0], result_refiner[0], structure[0], appearance[0]


@torch.no_grad()
def main(args):
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
    refiner_id_or_path = "stabilityai/stable-diffusion-xl-refiner-1.0"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    variant = "fp16" if device == "cuda" else "fp32"

    scheduler = RichControlDDIMScheduler.from_config(model_id_or_path, subfolder="scheduler")  # TODO: Support schedulers beyond DDIM
    refiner_scheduler = DDIMScheduler.from_config(scheduler.config)

    if args.model is None:
        pipe = RichControlStableDiffusionXLPipeline.from_pretrained(
            model_id_or_path, scheduler=scheduler, torch_dtype=torch_dtype, variant=variant, use_safetensors=True,
        )
    else:
        print(f"Using weights {args.model} for SDXL base model.")
        pipe = RichControlStableDiffusionXLPipeline.from_single_file(args.model, scheduler=scheduler, torch_dtype=torch_dtype)
    
    refiner = None
    if not args.disable_refiner:
        refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            refiner_id_or_path, scheduler=refiner_scheduler, text_encoder_2=pipe.text_encoder_2, vae=pipe.vae,
            torch_dtype=torch_dtype, variant=variant, use_safetensors=True,
        )
    
    if args.model_offload or args.sequential_offload:
        try:
            import accelerate  # Checking if accelerate is installed for Model/CPU offloading
        except:
            raise ModuleNotFoundError("`accelerate` must be installed for Model/CPU offloading.")
        
        if args.sequential_offload:
            pipe.enable_sequential_cpu_offload()
            if refiner is not None:
                refiner.enable_sequential_cpu_offload()
        elif args.model_offload:
            pipe.enable_model_cpu_offload()
            if refiner is not None:
                refiner.enable_model_cpu_offload()
    
    else:
        pipe = pipe.to(device)
        if refiner is not None:
            refiner = refiner.to(device)

    model_load_print = "Base model "
    if not args.disable_refiner:
        model_load_print += "+ refiner "
    if args.sequential_offload:
        model_load_print += "loaded with sequential CPU offloading."
    elif args.model_offload:
        model_load_print += "loaded with model CPU offloading."
    else:
        model_load_print += "loaded."
    print(f"{model_load_print} Running on device: {device}.")
    
    t = time()

    image_config = OmegaConf.load(args.image_config)
    model_config = OmegaConf.load(args.model_config)

    base_output_path = args.output_folder
    os.makedirs(base_output_path, exist_ok=True)

    image_output_path = args.output_folder
    makedirs(image_output_path, exist_ok=True)

    prompt = image_config['prompt']
    structure_prompt = image_config['inversion_prompt']
    structure_image = load_image(image_config['condition_image'])
    structure_image_original = deepcopy(structure_image)
    
    if model_config.get("erode", False):
        structure_image = erode_image(structure_image)
    if model_config.get("unsharp", False):
        structure_image = unsharp_mask_image(structure_image)

    appearance_prompt = ""
    if "appearance_prompt" in image_config:
        appearance_prompt = image_config['appearance_prompt']
    appearance_image = None
    if "appearance_image" in image_config:
        appearance_image = load_image(image_config['appearance_image'])

    structure_schedule = args.structure_schedule
    appearance_schedule = args.appearance_schedule

    if model_config.get('modify_prompt', False):
        appearance_prompt = compose_prompt(
            client_type="openai",
            api_key=model_config['prompt_model_api'],
            base_url=model_config['prompt_model_url'],
            model_name=model_config['prompt_model_name'],
            raw_prompt=prompt,
            image_path=image_config['condition_image']
        )
        appearance_prompt += " Best quality."
        
    result, result_refiner, structure, appearance = inference(
        pipe = pipe,
        refiner = refiner,
        device = device,
        structure_image = structure_image,
        appearance_image = appearance_image,
        prompt = prompt,
        structure_prompt = structure_prompt,
        appearance_prompt = appearance_prompt,
        positive_prompt = args.positive_prompt,
        negative_prompt = args.negative_prompt,
        guidance_scale = args.guidance_scale,
        structure_guidance_scale = args.structure_guidance_scale,
        appearance_guidance_scale = args.appearance_guidance_scale,
        num_inference_steps = args.num_inference_steps,
        eta = args.eta,
        seed = args.seed,
        width = args.width,
        height = args.height,
        structure_schedule = structure_schedule,
        appearance_schedule = appearance_schedule,
        model_config = model_config,
    )

    if result_refiner is None:
        result_refiner = result

    structure.save(path.join(image_output_path, "structure.png"))
    appearance.save(path.join(image_output_path, "appearance.png"))
    result_refiner.save(path.join(image_output_path, "result.png"))

    if args.benchmark:
        inference_time = time() - t
        peak_memory_usage = torch.cuda.max_memory_reserved()
        print(f"Inference time: {inference_time:.2f}s")
        print(f"Peak memory usage: {peak_memory_usage / pow(1024, 3):.2f}GiB")

    print("Done.")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--image_config", "-ic", type=str, default="./configs/image_config.yaml")
    parser.add_argument("--model_config", "-rc", type=str, default="./configs/model_config.yaml")
    parser.add_argument("--output_folder", "-o", type=str, default="./results")

    parser.add_argument("--structure_schedule", "-ss", type=float, default=0.6)
    parser.add_argument("--appearance_schedule", "-as", type=float, default=0.6)

    parser.add_argument("--positive_prompt", "-pp", type=str, default="high quality")
    parser.add_argument("--negative_prompt", "-np", type=str, default="ugly, blurry, dark, low res, unrealistic")

    parser.add_argument("--guidance_scale", "-g", type=float, default=5.0)
    parser.add_argument("--structure_guidance_scale", "-sg", type=float, default=5.0)
    parser.add_argument("--appearance_guidance_scale", "-ag", type=float, default=5.0)

    parser.add_argument("--num_inference_steps", "-n", type=int, default=50)
    parser.add_argument("--eta", "-e", type=float, default=1.0)

    parser.add_argument("--width", "-W", type=int, default=1024)
    parser.add_argument("--height", "-H", type=int, default=1024)

    parser.add_argument("--seed", "-s", type=int, default=3)

    parser.add_argument(
        "-mo", "--model_offload", action="store_true",
        help="Model CPU offload, lowers memory usage with slight runtime increase. `accelerate` must be installed.",
    )
    parser.add_argument(
        "-so", "--sequential_offload", action="store_true",
        help=(
            "Sequential layer CPU offload, significantly lowers memory usage with massive runtime increase."
            "`accelerate` must be installed. If both model_offload and sequential_offload are set, then use the latter."
        ),
    )
    parser.add_argument("-r", "--disable_refiner", action="store_true")
    parser.add_argument("-m", "--model", type=str, default=None, help="Optionally, load model safetensors.")
    parser.add_argument("-b", "--benchmark", action="store_true", help="Show inference time and max memory usage.")

    args = parser.parse_args()
    main(args)
