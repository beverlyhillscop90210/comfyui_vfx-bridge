"""
ComfyUI VFX Bridge - Node Definitions
"""

import os
import glob
import hashlib
import json
import torch
import numpy as np

# Try to import OpenEXR - will be needed for EXR loading
try:
    import OpenEXR
    import Imath
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False
    print("[VFX Bridge] Warning: OpenEXR not installed. Run: pip install OpenEXR Imath")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_latest_exr(folder_path: str) -> str | None:
    """Find the most recently modified EXR file in a folder."""
    if not os.path.isdir(folder_path):
        return None
    
    exr_files = glob.glob(os.path.join(folder_path, "*.exr"))
    exr_files += glob.glob(os.path.join(folder_path, "*.EXR"))
    
    if not exr_files:
        return None
    
    # Sort by modification time, newest first
    exr_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return exr_files[0]


def get_file_hash(filepath: str) -> str:
    """Get hash of file for change detection."""
    if not os.path.exists(filepath):
        return ""
    
    # Use modification time + size for fast change detection
    stat = os.stat(filepath)
    return f"{stat.st_mtime}_{stat.st_size}"


def load_exr_file(filepath: str) -> tuple[np.ndarray, dict, list[str]]:
    """
    Load an EXR file and return image data, metadata, and channel list.
    
    Returns:
        - image_data: numpy array [H, W, C] in float32
        - metadata: dict with resolution, bitdepth, colorspace, etc.
        - channels: list of channel names
    """
    if not HAS_OPENEXR:
        raise RuntimeError("OpenEXR not installed. Run: pip install OpenEXR Imath")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"EXR file not found: {filepath}")
    
    # Open the EXR file
    exr_file = OpenEXR.InputFile(filepath)
    header = exr_file.header()
    
    # Get data window (image dimensions)
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    # Get all channel names
    channels = list(header['channels'].keys())
    
    # Determine pixel type (16-bit half or 32-bit float)
    first_channel = header['channels'][channels[0]]
    if first_channel.type == Imath.PixelType(Imath.PixelType.HALF):
        pixel_type = Imath.PixelType(Imath.PixelType.HALF)
        np_dtype = np.float16
        bitdepth = 16
    else:
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        np_dtype = np.float32
        bitdepth = 32
    
    # Read all channels
    channel_data = {}
    for ch_name in channels:
        raw_data = exr_file.channel(ch_name, pixel_type)
        channel_data[ch_name] = np.frombuffer(raw_data, dtype=np_dtype).reshape(height, width)
    
    exr_file.close()
    
    # Extract metadata
    metadata = {
        "resolution": (width, height),
        "bitdepth": bitdepth,
        "channels": channels,
        "source_file": filepath,
        "filename": os.path.basename(filepath),
    }
    
    # Try to extract colorspace from header
    if 'chromaticities' in header:
        metadata["chromaticities"] = str(header['chromaticities'])
    
    # Check for common metadata keys
    for key in ['framesPerSecond', 'owner', 'comments', 'capDate', 'utcOffset']:
        if key in header:
            metadata[key] = str(header[key])
    
    # Try to get framerate
    if 'framesPerSecond' in header:
        fps = header['framesPerSecond']
        metadata['framerate'] = float(fps.n) / float(fps.d) if hasattr(fps, 'n') else float(fps)
    else:
        metadata['framerate'] = None
    
    # Organize channels into layers
    # Standard channels: R, G, B, A
    # Matte channels: matte.R, crypto.R, etc.
    
    # Build main RGBA image if possible
    rgba_channels = ['R', 'G', 'B', 'A']
    if all(ch in channel_data for ch in ['R', 'G', 'B']):
        if 'A' in channel_data:
            image_data = np.stack([
                channel_data['R'],
                channel_data['G'],
                channel_data['B'],
                channel_data['A']
            ], axis=-1).astype(np.float32)
        else:
            image_data = np.stack([
                channel_data['R'],
                channel_data['G'],
                channel_data['B']
            ], axis=-1).astype(np.float32)
    else:
        # Just stack all channels
        image_data = np.stack(list(channel_data.values()), axis=-1).astype(np.float32)
    
    # Store raw channel data in metadata for matte extraction
    metadata['_channel_data'] = {k: v.astype(np.float32) for k, v in channel_data.items()}
    
    return image_data, metadata, channels


# =============================================================================
# EXR HOT FOLDER LOADER NODE
# =============================================================================

class EXRHotFolderLoader:
    """
    Watches a folder and loads the latest EXR file automatically.
    Perfect for Nuke/Houdini render output integration.
    """
    
    CATEGORY = "VFX Bridge"
    FUNCTION = "load_exr"
    RETURN_TYPES = ("IMAGE", "VFX_METADATA", "STRING", "STRING")
    RETURN_NAMES = ("image", "metadata", "channels_info", "filename")
    OUTPUT_NODE = False
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "/path/to/exr/folder"
                }),
            },
            "optional": {
                "auto_refresh": ("BOOLEAN", {"default": True}),
            }
        }
    
    @classmethod
    def IS_CHANGED(cls, folder_path, auto_refresh=True):
        """Check if the latest EXR has changed."""
        if not auto_refresh:
            return False
        
        latest_exr = get_latest_exr(folder_path)
        if latest_exr:
            return get_file_hash(latest_exr)
        return ""
    
    def load_exr(self, folder_path: str, auto_refresh: bool = True):
        """Load the latest EXR from the folder."""
        
        # Find latest EXR
        latest_exr = get_latest_exr(folder_path)
        
        if latest_exr is None:
            raise ValueError(f"No EXR files found in: {folder_path}")
        
        # Load the EXR
        image_data, metadata, channels = load_exr_file(latest_exr)
        
        # Convert to torch tensor [B, H, W, C]
        image_tensor = torch.from_numpy(image_data).unsqueeze(0)
        
        # Create channel info string
        channels_info = ", ".join(channels)
        
        # Remove internal channel data from metadata for display
        metadata_display = {k: v for k, v in metadata.items() if not k.startswith('_')}
        
        # Store full metadata as JSON string for passing to other nodes
        # Keep _channel_data separate for matte splitter
        metadata_with_channels = metadata.copy()
        
        return (image_tensor, metadata_with_channels, channels_info, metadata['filename'])


# =============================================================================
# MATTE CHANNEL SPLITTER NODE
# =============================================================================

class MatteChannelSplitter:
    """
    Splits an EXR into individual matte channels.
    Each channel becomes a separate MASK output.
    """
    
    CATEGORY = "VFX Bridge"
    FUNCTION = "split_channels"
    OUTPUT_NODE = False
    
    # Dynamic outputs - we'll return up to 16 channels
    MAX_CHANNELS = 16
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "metadata": ("VFX_METADATA",),
            },
            "optional": {
                "channel_filter": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "e.g., matte, crypto (comma separated)"
                }),
            }
        }
    
    # We'll output up to 16 masks plus a list of channel names
    RETURN_TYPES = tuple(["MASK"] * MAX_CHANNELS + ["STRING"])
    RETURN_NAMES = tuple([f"channel_{i}" for i in range(MAX_CHANNELS)] + ["channel_names"])
    
    def split_channels(self, metadata: dict, channel_filter: str = ""):
        """Split EXR channels into individual mask outputs."""
        
        if '_channel_data' not in metadata:
            raise ValueError("No channel data in metadata. Connect to EXR Hot Folder Loader.")
        
        channel_data = metadata['_channel_data']
        all_channels = list(channel_data.keys())
        
        # Filter channels if specified
        if channel_filter.strip():
            filters = [f.strip().lower() for f in channel_filter.split(',')]
            filtered_channels = [ch for ch in all_channels 
                               if any(f in ch.lower() for f in filters)]
        else:
            filtered_channels = all_channels
        
        # Limit to MAX_CHANNELS
        selected_channels = filtered_channels[:self.MAX_CHANNELS]
        
        # Create mask outputs
        outputs = []
        for i in range(self.MAX_CHANNELS):
            if i < len(selected_channels):
                ch_name = selected_channels[i]
                # Convert to torch tensor [H, W] for MASK
                mask_data = channel_data[ch_name]
                mask_tensor = torch.from_numpy(mask_data)
                outputs.append(mask_tensor)
            else:
                # Return empty mask for unused slots
                if selected_channels:
                    h, w = channel_data[selected_channels[0]].shape
                else:
                    h, w = 512, 512  # Default size
                outputs.append(torch.zeros(h, w))
        
        # Add channel names as final output
        channel_names = ", ".join(selected_channels) if selected_channels else "No channels"
        outputs.append(channel_names)
        
        return tuple(outputs)


# =============================================================================
# METADATA DISPLAY NODE
# =============================================================================

class MetadataDisplay:
    """
    Displays EXR metadata in the ComfyUI interface.
    Shows resolution, framerate, bitdepth, colorspace.
    """
    
    CATEGORY = "VFX Bridge"
    FUNCTION = "display_metadata"
    RETURN_TYPES = ("STRING", "INT", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("info_text", "width", "height", "bitdepth", "framerate", "colorspace")
    OUTPUT_NODE = True  # This is a display node
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "metadata": ("VFX_METADATA",),
            },
        }
    
    def display_metadata(self, metadata: dict):
        """Extract and display metadata."""
        
        # Get resolution
        resolution = metadata.get('resolution', (0, 0))
        width, height = resolution
        
        # Get other metadata
        bitdepth = metadata.get('bitdepth', 16)
        framerate = metadata.get('framerate') or 0.0
        colorspace = metadata.get('chromaticities', 'Unknown')
        filename = metadata.get('filename', 'Unknown')
        channels = metadata.get('channels', [])
        
        # Build info text
        info_lines = [
            f"📁 File: {filename}",
            f"📐 Resolution: {width} × {height}",
            f"🎨 Bit Depth: {bitdepth}-bit",
            f"🎬 Framerate: {framerate:.2f} fps" if framerate else "🎬 Framerate: N/A",
            f"🌈 Colorspace: {colorspace}",
            f"📊 Channels ({len(channels)}): {', '.join(channels[:8])}{'...' if len(channels) > 8 else ''}"
        ]
        info_text = "\n".join(info_lines)
        
        # Also print to console for visibility
        print("\n" + "=" * 50)
        print("VFX Bridge - EXR Metadata")
        print("=" * 50)
        for line in info_lines:
            print(line)
        print("=" * 50 + "\n")
        
        return (info_text, width, height, bitdepth, float(framerate or 0.0), str(colorspace))


# =============================================================================
# EXR SAVE NODE
# =============================================================================

class EXRSaveNode:
    """
    Saves images/mattes back to a 16-bit EXR file.
    Preserves metadata from the original file.
    """
    
    CATEGORY = "VFX Bridge"
    FUNCTION = "save_exr"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "output_folder": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "/path/to/output/folder"
                }),
                "filename": ("STRING", {
                    "default": "output",
                    "multiline": False,
                }),
            },
            "optional": {
                "metadata": ("VFX_METADATA",),
                "bitdepth": (["16", "32"], {"default": "16"}),
            }
        }
    
    def save_exr(self, image: torch.Tensor, output_folder: str, filename: str, 
                 metadata: dict = None, bitdepth: str = "16"):
        """Save image to EXR file."""
        
        if not HAS_OPENEXR:
            raise RuntimeError("OpenEXR not installed. Run: pip install OpenEXR Imath")
        
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Prepare filename
        if not filename.endswith('.exr'):
            filename = f"{filename}.exr"
        output_path = os.path.join(output_folder, filename)
        
        # Convert tensor to numpy [B, H, W, C] -> [H, W, C]
        if image.dim() == 4:
            image_np = image[0].cpu().numpy()  # Take first image from batch
        else:
            image_np = image.cpu().numpy()
        
        height, width = image_np.shape[:2]
        num_channels = image_np.shape[2] if image_np.ndim == 3 else 1
        
        # Determine pixel type
        if bitdepth == "16":
            pixel_type = Imath.PixelType(Imath.PixelType.HALF)
            np_dtype = np.float16
        else:
            pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
            np_dtype = np.float32
        
        # Prepare header
        header = OpenEXR.Header(width, height)
        
        # Set up channels
        if num_channels >= 3:
            channel_names = ['R', 'G', 'B']
            if num_channels >= 4:
                channel_names.append('A')
        else:
            channel_names = ['Y']  # Grayscale
        
        # Create channel dict for header
        header['channels'] = {
            name: Imath.Channel(pixel_type) for name in channel_names
        }
        
        # Add metadata from original if available
        if metadata:
            if 'chromaticities' in metadata:
                # Note: Would need proper chromaticities object
                pass
            if 'comments' in metadata:
                header['comments'] = metadata['comments']
        
        # Prepare channel data
        channel_data = {}
        for i, name in enumerate(channel_names):
            if i < num_channels:
                channel_array = image_np[:, :, i].astype(np_dtype)
            else:
                channel_array = np.zeros((height, width), dtype=np_dtype)
            channel_data[name] = channel_array.tobytes()
        
        # Write EXR file
        exr_file = OpenEXR.OutputFile(output_path, header)
        exr_file.writePixels(channel_data)
        exr_file.close()
        
        print(f"[VFX Bridge] Saved EXR: {output_path}")
        
        return (output_path,)


# =============================================================================
# PREVIEW CHANNEL NODE (for individual matte preview)
# =============================================================================

class PreviewMatte:
    """
    Preview a single matte channel as an image.
    Converts MASK to IMAGE for preview nodes.
    """
    
    CATEGORY = "VFX Bridge"
    FUNCTION = "preview_matte"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)
    OUTPUT_NODE = False
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "matte": ("MASK",),
            },
            "optional": {
                "colorize": ("BOOLEAN", {"default": False}),
                "color": (["red", "green", "blue", "white"], {"default": "white"}),
            }
        }
    
    def preview_matte(self, matte: torch.Tensor, colorize: bool = False, color: str = "white"):
        """Convert mask to previewable image."""
        
        # Ensure 2D tensor [H, W]
        if matte.dim() == 3:
            matte = matte[0]  # Take first if batched
        
        h, w = matte.shape
        
        if colorize:
            # Create colored matte
            color_map = {
                "red": [1.0, 0.0, 0.0],
                "green": [0.0, 1.0, 0.0],
                "blue": [0.0, 0.0, 1.0],
                "white": [1.0, 1.0, 1.0],
            }
            rgb = color_map.get(color, [1.0, 1.0, 1.0])
            
            image = torch.zeros(1, h, w, 3)
            for i, c in enumerate(rgb):
                image[0, :, :, i] = matte * c
        else:
            # Grayscale preview
            image = matte.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)
        
        return (image,)


# =============================================================================
# NODE MAPPINGS
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "EXRHotFolderLoader": EXRHotFolderLoader,
    "MatteChannelSplitter": MatteChannelSplitter,
    "MetadataDisplay": MetadataDisplay,
    "EXRSaveNode": EXRSaveNode,
    "PreviewMatte": PreviewMatte,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EXRHotFolderLoader": "EXR Hot Folder Loader",
    "MatteChannelSplitter": "Matte Channel Splitter", 
    "MetadataDisplay": "Metadata Display",
    "EXRSaveNode": "EXR Save",
    "PreviewMatte": "Preview Matte",
}
