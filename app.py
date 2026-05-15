import os
import sys
import yaml
import torch
import logging
import traceback
import gradio as gr
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Mol-App")



from model.train_loop import TrainLoop
from utils.build_mol import MoleculeBuilder
from lldm.diffusion import LinkerLatentDiffusion
from lldm.regressor import NumAtomsRegressor

# ==========================================
# predefined prompts 
# ==========================================
PRESET_PROMPTS = [
    "A lipophilic molecule with a complex ring system combining saturated and aromatic parts with aromatic rings and a single aliphatic ring. Notable functional groups include halogen substituents, sulfonyl, and a amide bond",
    "A low-molecular-weight, non-polar chemical. Additionally, it features ether linkage and a single secondary amine",
    "A fat-soluble target with a fully aromatic framework containing 4 aromatic rings. Notable functional groups include a single ether linkage and amide bonds.",
    "The proposed structure is a inflexible, small compound characterized by its hybrid cyclic system containing a single benzene ring, and is substituted with halogen substituents, ether linkages, and primary amine",
    "A moderately sized, greasy compound with a complex ring system combining saturated and aromatic parts that includes a single thiophene ring and 1 benzene ring. Notable functional groups include an primary amine and halogen substituents."
]

# global model container, only loaded once at startup and shared across all requests
class AppModels:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.text_encoder = None
        self.diffusion_model = None
        self.regressor = None
        self.vae_model = None
        self.builder = None
        self.inv_atom_map = None

    def load_all(self):
        logger.info(f"Using device: {self.device}")
        
        # 1.  BERT
        logger.info("Loading SentenceTransformer...")
        self.text_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
        
        # 2.  LLDM Diffusion
        diff_ckpt_path = os.path.join(ckpt_dir, "lldm.pt")
        logger.info(f"Loading Diffusion Model from {diff_ckpt_path}...")
        diff_ckpt = torch.load(diff_ckpt_path, map_location='cpu')
        config = diff_ckpt['config']
        if 'text_emb_dim' not in config: config['text_emb_dim'] = 384
        self.diffusion_model = LinkerLatentDiffusion(config).to(self.device)
        self.diffusion_model.load_state_dict(diff_ckpt.get('ema_state', diff_ckpt['model_state']))
        self.diffusion_model.eval()
        
        # 3.  Regressor
        reg_ckpt_path = os.path.join(ckpt_dir, "regressor.pt")
        logger.info(f"Loading NumAtoms Regressor from {reg_ckpt_path}...")
        reg_ckpt = torch.load(reg_ckpt_path, map_location='cpu')
        self.regressor = NumAtomsRegressor(reg_ckpt['config']).to(self.device)
        self.regressor.load_state_dict(reg_ckpt['model_state'])
        self.regressor.eval()
        
        # 4.  VAE Decoder
        vae_ckpt_path = os.path.join(ckpt_dir, "vae.ckpt")
        vae_config_path = os.path.join(BASE_DIR, "config", "config.yaml")
        logger.info(f"Loading LinkerVAE Decoder from {vae_ckpt_path}...")
        with open(vae_config_path, 'r') as f:
            vae_cfg = yaml.safe_load(f)
            
        atom_map = {6: 0, 7: 1, 8: 2, 9: 3, 15: 4, 16: 5, 17: 6, 35: 7, 53: 8} # Default
        vocab_size = 9
        self.inv_atom_map = {v: k for k, v in atom_map.items()}
        
        vae_cfg['encoder_config']['ligand_v_dim'] = vocab_size
        vae_cfg['decoder_config']['ligand_atom_feature_dim'] = vocab_size
        
        vae_ckpt = torch.load(vae_ckpt_path, map_location='cpu', weights_only=False)
        self.vae_model = TrainLoop(vae_cfg)
        self.vae_model.load_state_dict(vae_ckpt['state_dict'], strict=True)
        self.vae_model.to(self.device)
        try:
            self.vae_model.decoder.device = self.device
            self.vae_model.decoder.sigma1_coord = self.vae_model.decoder.sigma1_coord.to(self.device)
            self.vae_model.decoder.beta1 = self.vae_model.decoder.beta1.to(self.device)
        except Exception: pass
        self.vae_model.eval()
        
        # molecule builder
        self.builder = MoleculeBuilder()
        logger.info("✅ All Models Loaded Successfully!")

# global model instance
app_models = AppModels()
app_models.load_all()



def render_3d_mol(evt: gr.SelectData, sdf_list):
    """
    when user clicks on a gallery image, this function will be triggered 
    with the index of the clicked image (evt.index) and the list of SDF strings (sdf_list) generated in the current session. 
    It will return an HTML string that embeds a 3Dmol.js viewer rendering the corresponding SDF structure. 
    If no valid SDF is found for the clicked index, it returns a placeholder message instead.
    """
    print(f"\n[DEBUG 3D] Gallery clicked! Image index: {evt.index}")
    print(f"[DEBUG 3D] Total SDFs in state: {len(sdf_list) if sdf_list else 0}")
    
    if not sdf_list or evt.index >= len(sdf_list):
        print("[DEBUG 3D] No SDF found for this index or list is empty!")
        return "<p style='text-align:center; color:gray;'>No 3D data available</p>"
    
    sdf_block = sdf_list[evt.index]
    print(f"[DEBUG 3D] Found SDF block! Length: {len(sdf_block)} chars.")
    snippet = sdf_block[:100].replace('\n', ' ')
    print(f"[DEBUG 3D] SDF Snippet: {snippet}...")

    # Jsx 3Dmol.js viewer embedded in an iframe, using srcdoc to directly inject the HTML content
    html_template = f"""
    <iframe style="width: 100%; height: 400px; border: 1px solid #e5e7eb; border-radius: 8px;" srcdoc="
        <!DOCTYPE html>
        <html>
        <head>
            <!-- 3Dmol.js -->
            <script src='https://3Dmol.csb.pitt.edu/build/3Dmol-min.js'></script>
        </head>
        <body style='margin:0; padding:0; overflow:hidden; background-color: #f9fafb;'>
            <div id='container' style='width: 100%; height: 100%; position: absolute;'></div>
            
            <script>
                // avoid potential CORS issues by using srcdoc and directly embedding the SDF data
                window.onload = function() {{
                    console.log('Iframe loaded, starting 3Dmol...');
                    try {{
                        let element = document.getElementById('container');
                        let viewer = $3Dmol.createViewer(element, {{backgroundColor: '#f9fafb'}});
                        let sdfData = `{sdf_block}`;
                        
                        viewer.addModel(sdfData, 'sdf');
                        viewer.setStyle({{}}, {{stick: {{radius: 0.15}}, sphere: {{scale: 0.3}}}});
                        viewer.zoomTo();
                        viewer.render();
                        console.log('3Dmol rendering complete!');
                    }} catch (e) {{
                        console.error('3Dmol failed:', e);
                    }}
                }};
            </script>
        </body>
        </html>
    "></iframe>
    """
    return html_template

# core generation function, takes in the prompt and parameters, returns generated images, status logs, sdf list, and 3D viewer HTML
@torch.no_grad()
def generate_and_draw(prompt, num_samples, guidance_scale, bfn_steps):
    status_logs = []
    generated_images = []
    generated_sdfs = [] 
    
    prompt = prompt.strip()
    
    # decide if it's conditional or unconditional generation
    is_unconditional = (len(prompt) == 0)
    
    try:
        if is_unconditional:
            condition_batch = None
            status_logs.append("Mode: Unconditional Generation") 
        else:
            status_logs.append(f"Mode: Conditional (Guidance: {guidance_scale})")
            emb = app_models.text_encoder.encode(prompt, convert_to_tensor=True, device=app_models.device)
            condition_batch = emb.unsqueeze(0).repeat(num_samples, 1)

        # dynamically set guidance scale in the diffusion model
        app_models.diffusion_model.config['guidance_scale'] = float(guidance_scale)

        # B. sampling latent representations (Zx, Zh) from LLDM
        status_logs.append(f"Sampling {num_samples} Latents (Diffusion)...")
        zx_gen, zh_gen = app_models.diffusion_model.sample(n_samples=num_samples, condition=condition_batch)

        # C. predicting number of atoms for each molecule using the regressor
        pred_n = app_models.regressor(zx_gen, zh_gen)
        pred_n = torch.clamp(torch.round(pred_n), min=1).long()

        # D. (Batch Decoding)
        status_logs.append(f"Batch Decoding 3D Structures (BFN) and rendering 2D...")
        
        # 1. flatten Zx and Zh for batch processing
        # [B, 10, dim] to [B*10, dim]
        flat_zx = zx_gen.view(-1, 3)
        flat_zh = zh_gen.view(-1, 32)
        
        # 2. index Tensor
        # global_batch is used to indicate which molecule each atom belongs to across the entire batch of samples.
        global_batch = torch.arange(num_samples, device=app_models.device).repeat_interleave(10)
        batch_ligand = torch.arange(num_samples, device=app_models.device).repeat_interleave(pred_n)

        # 3. batch decode using BFN decoder
        with torch.no_grad():
            _, sample_chain, _ = app_models.vae_model.decoder.sample(
                protein_pos=flat_zx,
                protein_v=flat_zh,
                batch_protein=global_batch,
                batch_ligand=batch_ligand,
                n_nodes=num_samples, 
                sample_steps=int(bfn_steps),
                desc=None
            )
            
        # get final predicted positions and atom types from the last sample in the chain
        final_sample = sample_chain[-1]
        pred_pos, recon_one_hot = final_sample[0], final_sample[1]
        recon_h_indices = recon_one_hot.argmax(dim=-1)

        # 4. split the flat predictions back into individual molecules and render
        for i in range(num_samples):
            try:
                mask = (batch_ligand == i)
                coords = pred_pos[mask].cpu().numpy()
                atom_indices = recon_h_indices[mask].cpu().tolist()
                atom_types = [app_models.inv_atom_map[idx] for idx in atom_indices]
                n_atoms = pred_n[i].item()
                
                # 3d mols
                recon_mol_3d = app_models.builder.build_mol(coords, atom_types)
                # export to sdf block for 3D viewer
                sdf_block = Chem.MolToMolBlock(recon_mol_3d)
                generated_sdfs.append(sdf_block)
                
                # create a 2D depiction for the gallery
                recon_mol_2d = Chem.Mol(recon_mol_3d)
                recon_mol_2d = Chem.RemoveHs(recon_mol_2d)
                AllChem.Compute2DCoords(recon_mol_2d) # this modifies recon_mol_2d in place to have 2D coordinates
                
                img = Draw.MolToImage(recon_mol_2d, size=(600, 600), fitImage=True)
                caption = f"Mol {i+1} | Atoms: {n_atoms}"
                generated_images.append((img, caption))
                
            except Exception as e:
                logger.error(f"Mol {i} failed: {str(e)}")
                continue

        torch.cuda.empty_cache()
        status_logs.append(f" Successfully generated {len(generated_images)} valid molecules.")
        
        # return the list of generated images with captions, the status logs, the list of SDF blocks for 3D rendering
        empty_3d_html = "<p style='text-align:center; color:gray; line-height:400px;'>Click an image above to view 3D structure</p>"
        print(f"[DEBUG Generate] Generated {len(generated_sdfs)} SDF blocks.")
        return generated_images, "\n".join(status_logs), generated_sdfs, empty_3d_html

    except Exception as e:
        traceback.print_exc()
        torch.cuda.empty_cache()
        return generated_images, f" Pipeline Failed:\n{str(e)}", [], ""

# gradio UI definition, including the new 3D viewer and the updated clear button functionality
def create_ui():
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="indigo",
        neutral_hue="slate"
    )
    
    with gr.Blocks(theme=theme, title="LLDM Text-to-Molecule") as app:
        gr.Markdown(
            """
            # 🧬 LLDM: Text-Conditioned Molecule Generation
            Enter a text prompt to generate molecule structures. Leave the prompt empty for **Unconditional Generation**.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                # input prompt
                prompt_input = gr.Textbox(
                    label="Text Prompt (Condition)", 
                    placeholder="E.g., A lipophilic molecule with a complex ring system...",
                    lines=4
                )
                
                # mode preset dropdown
                preset_dropdown = gr.Dropdown(
                    choices=PRESET_PROMPTS, 
                    label="Or select a preset prompt:",
                    value=None
                )
                
                # clear button (optimized to clear both Textbox and Dropdown)
                clear_btn = gr.Button("🗑️ Clear Prompt Text (Unconditional Mode)", size="sm")
                
                clear_btn.click(
                    fn=lambda: ("", None), 
                    inputs=None, 
                    outputs=[prompt_input, preset_dropdown]
                )
                
                def update_prompt(preset): return preset
                preset_dropdown.change(fn=update_prompt, inputs=preset_dropdown, outputs=prompt_input)
                
                with gr.Accordion("Advanced Settings", open=True):
                    num_samples_slider = gr.Slider(minimum=1, maximum=40, value=9, step=1, label="Number of Molecules")
                    guidance_scale_slider = gr.Slider(minimum=1.0, maximum=10.0, value=7.0, step=0.5, label="Guidance Scale (CFG)")
                    bfn_steps_slider = gr.Slider(minimum=40, maximum=200, value=80, step=10, label="BFN Denoising Steps")
                
                generate_btn = gr.Button("🚀 Generate Molecules", variant="primary", size="lg")
                status_output = gr.Textbox(label="Status Logs", interactive=False, lines=5)

                # qr code image for ngrok sharing (not used after presentation, but kept for reference)
                gr.Image(
                    value="qrcode.jpg",
                    label="QR Code for ngrok Sharing (no longer used)",
                    interactive=False,    
                    show_label=False,    
                    show_download_button=False 
                )

            with gr.Column(scale=3):
                gallery_output = gr.Gallery(
                    label="Generated Molecules (2D)", 
                    show_label=True, 
                    elem_id="gallery", 
                    columns=[3], 
                    rows=[2], 
                    object_fit="contain", 
                    height="auto"
                )
                
                # 3D viewer area
                gr.Markdown("### Interactive 3D Viewer")
                html_3d_viewer = gr.HTML(
                    value="<p style='text-align:center; color:gray; line-height:400px;'>Click an image above to view 3D structure</p>", 
                    elem_id="mol3d"
                )
                

        sdf_state = gr.State([])

        generate_btn.click(
            fn=generate_and_draw,
            inputs=[prompt_input, num_samples_slider, guidance_scale_slider, bfn_steps_slider],
            outputs=[gallery_output, status_output, sdf_state, html_3d_viewer]
        )
        
        # Gallery click event to render 3D structure
        gallery_output.select(
            fn=render_3d_mol,
            inputs=[sdf_state],
            outputs=[html_3d_viewer],
            queue=False  # no need to queue this, we want it to be responsive and it doesn't involve heavy computation
        )
        
    return app

### Main entry point ###

if __name__ == "__main__":
    app = create_ui()
    
    # to avoid some gradio bug
    import os
    os.environ['no_proxy'] = 'localhost,127.0.0.1,0.0.0.0'
    os.environ['HTTP_PROXY'] = ''
    os.environ['HTTPS_PROXY'] = ''


    app.launch(
        server_name="0.0.0.0", 
        server_port=7862, 
        share=False, 
        show_api=False,
        debug=True
    )