"""CAD-Coder Gradio UI with subprocess CAD export (not a hardened sandbox)."""
import os
import re
import subprocess
import uuid
from pathlib import Path
from threading import Lock

import gradio as gr
import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images

MODEL_PATH = os.getenv('MODEL_PATH', '/workspace/models/CAD-Coder')
OUT = Path('/workspace/outputs')
OUT.mkdir(parents=True, exist_ok=True)
LOCK = Lock()

# Keep a single GPU model loaded across requests.
tokenizer, model, image_processor, context_len = load_pretrained_model(
    MODEL_PATH, None, 'cad-coder', load_4bit=os.getenv('LOAD_4BIT', '1') == '1',
    device_map='auto', use_flash_attn=False,
)
model.eval()


def convert(job_dir: Path, py_path: Path):
    """Execute geometry generation as unprivileged cadexec with timeout and limits.

    The process has its own UID and limited resources but retains container filesystem
    and networking capabilities. DO NOT use this for untrusted/public submissions.
    """
    step = job_dir / 'design.step'
    preview = job_dir / 'preview.stl'
    cmd = [
        '/usr/bin/setpriv', '--reuid=cadexec', '--regid=cadexec', '--clear-groups',
        '/usr/bin/prlimit', '--cpu=90:90', '--as=8589934592:8589934592',
        '--fsize=268435456:268435456', '--nproc=64:64',
        '/opt/conda/envs/cad/bin/python', '/opt/app/cad_converter.py',
        str(py_path), str(step), str(preview),
    ]
    try:
        proc = subprocess.run(cmd, cwd=str(job_dir), capture_output=True, text=True,
                              timeout=120, env={'PATH': '/usr/bin:/bin', 'HOME': str(job_dir),
                                                'PYTHONNOUSERSITE': '1', 'OMP_NUM_THREADS': '2'})
    except subprocess.TimeoutExpired:
        return None, None, 'CAD conversion timed out after 120 seconds.'
    if proc.returncode != 0:
        details = (proc.stderr or proc.stdout or 'Unknown conversion failure')[-1800:]
        return None, None, 'STEP conversion failed. Check generated CadQuery code.\n' + details
    if not step.exists() or not preview.exists():
        return None, None, 'Conversion finished without STEP and STL output.'
    return str(step), str(preview), 'STEP and 3D preview created.'


def generate(image, instruction, max_tokens):
    if image is None:
        raise gr.Error('Upload a reference image first.')
    if not torch.cuda.is_available():
        raise gr.Error('CUDA GPU not detected.')
    question = (instruction or 'Generate complete executable CadQuery Python code for this object.').strip()
    if model.config.mm_use_im_start_end:
        question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
    else:
        question = DEFAULT_IMAGE_TOKEN + '\n' + question
    conv = conv_templates['llava_v1'].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to('cuda')
    pixels = process_images([image.convert('RGB')], image_processor, model.config)[0]
    with LOCK, torch.inference_mode():
        result = model.generate(ids, images=pixels.unsqueeze(0).half().to('cuda'),
                                image_sizes=[image.size], do_sample=False,
                                max_new_tokens=int(max_tokens), use_cache=True)
    # Upstream LLaVA convention: generated IDs do not contain the input prefix.
    if result.shape[1] >= ids.shape[1] and torch.equal(result[0, :ids.shape[1]], ids[0]):
        result = result[:, ids.shape[1]:]
    response = tokenizer.batch_decode(result, skip_special_tokens=True)[0].strip()
    blocks = re.findall(r'```(?:python)?\s*(.*?)```', response, flags=re.S | re.I)
    code = blocks[0].strip() if blocks else response
    job = OUT / uuid.uuid4().hex
    job.mkdir(parents=True, exist_ok=False)
    os.chmod(job, 0o770)
    # The converter only has write access to this output folder, via cadexec group.
    import grp
    os.chown(job, -1, grp.getgrnam('cadexec').gr_gid)
    py_path = job / 'generated.py'
    py_path.write_text(code + '\n', encoding='utf-8')
    os.chmod(py_path, 0o640)
    os.chown(py_path, -1, grp.getgrnam('cadexec').gr_gid)
    step, preview, status = convert(job, py_path)
    return code, str(py_path), step, preview, status


with gr.Blocks(title='CAD-Coder / Vast.ai') as app:
    gr.Markdown('# CAD-Coder · Image to STEP\nUpload an image and generate CadQuery code, a STEP solid, and an STL 3D preview. **Run locally through your SSH tunnel only.** Generated code is executed in a restricted subprocess, not a complete security sandbox.')
    with gr.Row():
        img = gr.Image(type='pil', label='Reference image')
        with gr.Column():
            prompt = gr.Textbox(label='CAD instructions', value='Generate complete executable CadQuery Python code for this object. Assign the final solid to model.', lines=4)
            tokens = gr.Slider(128, 2048, value=1024, step=128, label='Maximum output tokens')
            run = gr.Button('Generate CAD + STEP', variant='primary')
    status = gr.Textbox(label='Conversion status', interactive=False)
    with gr.Row():
        code = gr.Code(label='Generated CadQuery Python', language='python')
        preview = gr.Model3D(label='3D preview (STL)', clear_color=[0.12, 0.12, 0.12, 1.0])
    with gr.Row():
        download_py = gr.File(label='Download CadQuery Python')
        download_step = gr.File(label='Download STEP model')
    run.click(generate, [img, prompt, tokens], [code, download_py, download_step, preview, status], concurrency_limit=1)

app.queue(default_concurrency_limit=1)
app.launch(server_name='127.0.0.1', server_port=7860, share=False, show_error=True,
           allowed_paths=[str(OUT)])
