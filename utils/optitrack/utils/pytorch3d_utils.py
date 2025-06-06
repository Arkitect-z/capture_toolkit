import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import imageio # Added for video writing
from tqdm import tqdm # Added for internal progress bar

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import (
    AxisArgs,
    plot_batch_individually,
    plot_scene,
)
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
)

# add path for demo utils functions
import sys
# sys.path.append(os.path.abspath("")) # Already handled in visualize.py

# Setup device (this global device might be overridden by passed-in device)
if torch.cuda.is_available():
    g_device = torch.device("cuda:0")
    torch.cuda.set_device(g_device)
else:
    g_device = torch.device("cpu")


def cal_camera_RT(mesh, device_to_use=None): # Added device_to_use
    # Compute the bounding box of the mesh
    # Ensure mesh is on the correct device
    current_device = device_to_use if device_to_use is not None else mesh.device
    verts = mesh.verts_packed().to(current_device)
    
    min_xyz = verts.min(0)[0]
    max_xyz = verts.max(0)[0]
    center = ((min_xyz + max_xyz) / 2)[None]
    size = max_xyz - min_xyz
    max_size = size.max().item()

    # Set the camera distance based on the size of the bounding box
    camera_distance = 1.5 * max_size  # Adjust this factor as needed

    # Set up the camera
    R, T = look_at_view_transform(
        dist=camera_distance, elev=20, azim=30, at=center.to(current_device)
    )

    return R.to(current_device), T.to(current_device)


def single_visualize(verts, faces, RT=None, save_path=None, device_to_use=None): # Added device_to_use
    # If device_to_use is not provided, use the global device
    current_device = device_to_use if device_to_use is not None else g_device
    
    # prepare data
    verts = verts.detach().to(current_device)
    faces = torch.tensor(faces.astype(np.int32) if isinstance(faces, np.ndarray) else faces, device=current_device) # Ensure faces is a tensor
    
    # get textures
    verts_rgb = torch.ones_like(verts)[None] * 0.7  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb.to(current_device))
    # get mesh
    mesh = Meshes(verts=[verts], faces=[faces], textures=textures)

    if RT == None:
        R, T = cal_camera_RT(mesh, device_to_use=current_device)
    else:
        R, T = RT[0].to(current_device), RT[1].to(current_device) # Ensure R, T are on current_device
        
    cameras = FoVPerspectiveCameras(device=current_device, R=R, T=T)

    raster_settings = RasterizationSettings(
        image_size=1024,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    lights = PointLights(device=current_device, location=[[0.0, 0.0, 3.0]]) # Adjusted light position

    materials = Materials(
        specular_color=((0.2, 0.2, 0.2),),
        shininess=30,
        device=current_device,
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=current_device, cameras=cameras, lights=lights, materials=materials
        ),
    )
    
    images = renderer(mesh, lights=lights) # Pass lights explicitly

    image_to_save = images[0, ..., :3].cpu().numpy()
    image_to_save = np.clip(image_to_save, 0, 1)
    image_to_save = (image_to_save * 255).astype(np.uint8)

    if save_path:
        plt.imsave(save_path, image_to_save)
    else:
        # Fallback to show if not saving (though not used by visualize.py anymore)
        plt.figure(figsize=(10, 10))
        plt.imshow(image_to_save)
        plt.axis("off")
        plt.show()


def sequence_visualize(all_vertices_list, # List of tensors, each [num_verts, 3]
                       faces_tensor,      # Tensor of faces [num_faces, 3]
                       output_video_path,
                       fps=30,
                       image_size=1024,
                       camera_rt=None,    # Optional precomputed R, T
                       device=None):      # Explicitly pass device

    if not all_vertices_list:
        print("Warning: No vertices provided for sequence visualization.")
        return

    # Determine device to use
    current_device = device if device is not None else (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
    print(f"Sequence visualizer using device: {current_device}")

    # Ensure faces_tensor is on the correct device
    faces_tensor = faces_tensor.to(current_device)

    # --- Initialize Renderer Components (once) ---
    if camera_rt is None:
        # Create a mesh from the first frame to calculate camera parameters
        # Ensure the first vertex tensor is on the correct device
        first_frame_verts = all_vertices_list[0].to(current_device)
        first_frame_mesh = Meshes(verts=[first_frame_verts], faces=[faces_tensor])
        R, T = cal_camera_RT(first_frame_mesh, device_to_use=current_device)
    else:
        R, T = camera_rt[0].to(current_device), camera_rt[1].to(current_device)
    
    cameras = FoVPerspectiveCameras(device=current_device, R=R, T=T)

    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    # Adjusted light position based on typical PyTorch3D coordinate system (+Y up, +X right, +Z out of screen)
    # Light in front and slightly above the object.
    lights = PointLights(device=current_device, location=[[0.0, 1.0, 3.0]]) 

    materials = Materials(
        specular_color=((0.2, 0.2, 0.2),),
        shininess=30,
        device=current_device,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=current_device, cameras=cameras, lights=lights, materials=materials),
    )

    print(f"Starting video generation for {output_video_path}...")
    # Using imageio to write the video
    # Added macro_block_size=1 which can help with some codecs if frame dimensions are not divisible by macro block size.
    with imageio.get_writer(output_video_path, fps=fps, macro_block_size=1) as video_writer:
        for verts_frame_original in tqdm(all_vertices_list, desc="Rendering video frames"):
            verts_frame = verts_frame_original.to(current_device) # Ensure current frame's vertices are on device
            
            # Create Textures for each frame. Assuming a constant color.
            verts_rgb = torch.ones_like(verts_frame)[None] * 0.7  # (1, V, 3)
            textures = TexturesVertex(verts_features=verts_rgb.to(current_device))

            mesh_frame = Meshes(
                verts=[verts_frame], 
                faces=[faces_tensor], # faces_tensor is already on current_device
                textures=textures
            )
            
            image_tensor = renderer(mesh_frame) # image_tensor shape (1, H, W, 4)
            
            # Convert to uint8 NumPy array
            image_np = image_tensor[0, ..., :3].cpu().numpy() # Drop alpha, move to CPU
            image_np_clipped = np.clip(image_np, 0, 1) # Ensure values are in [0,1] before scaling
            image_uint8 = (image_np_clipped * 255).astype(np.uint8)
            
            video_writer.append_data(image_uint8)
            
    print(f"Video saved successfully to {output_video_path}")