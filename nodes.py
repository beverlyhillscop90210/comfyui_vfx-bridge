"""
ComfyUI VFX Bridge - Node Definitions
Full OCIO (OpenColorIO) Integration for VFX Pipelines
"""

import os
import glob
import torch
import numpy as np

# Try to import OpenEXR
try:
    import OpenEXR
    import Imath
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False
    print("[VFX Bridge] Warning: OpenEXR not installed. Run: pip install OpenEXR Imath")

# Try to import OpenColorIO
try:
    import PyOpenColorIO as OCIO
    HAS_OCIO = True
    OCIO_VERSION = OCIO.__version__
    print(f"[VFX Bridge] OCIO {OCIO_VERSION} loaded successfully")
except ImportError:
    HAS_OCIO = False
    OCIO_VERSION = None
    print("[VFX Bridge] OCIO not installed - using built-in color transforms")


# =============================================================================
# COLOR TRANSFORM MATRICES (Fallback when no OCIO config)
# =============================================================================

# sRGB to Linear
def srgb_to_linear(img):
    """sRGB to Linear transform."""
    threshold = 0.04045
    return np.where(img <= threshold, img / 12.92, 
                    np.power((img + 0.055) / 1.055, 2.4))

def linear_to_srgb(img):
    """Linear to sRGB transform."""
    threshold = 0.0031308
    return np.where(img <= threshold, img * 12.92,
                    1.055 * np.power(np.clip(img, threshold, None), 1/2.4) - 0.055)

# Rec.709
def linear_to_rec709(img):
    """Linear to Rec.709 OETF."""
    threshold = 0.018
    return np.where(img < threshold, img * 4.5,
                    1.099 * np.power(np.clip(img, threshold, None), 0.45) - 0.099)

def rec709_to_linear(img):
    """Rec.709 to Linear."""
    threshold = 0.081
    return np.where(img < threshold, img / 4.5,
                    np.power((img + 0.099) / 1.099, 1/0.45))

# ACEScg to sRGB (simplified - AP1 to sRGB primaries + gamma)
ACESCG_TO_SRGB_MATRIX = np.array([
    [1.70505, -0.62179, -0.08326],
    [-0.13026,  1.14080, -0.01055],
    [-0.02400, -0.12897,  1.15297]
])

SRGB_TO_ACESCG_MATRIX = np.array([
    [0.61309, 0.33952, 0.04737],
    [0.07019, 0.91635, 0.01345],
    [0.02062, 0.10957, 0.86961]
])

def acescg_to_linear_srgb(img):
    """ACEScg to Linear sRGB primaries."""
    shape = img.shape
    flat = img.reshape(-1, 3)
    result = flat @ ACESCG_TO_SRGB_MATRIX.T
    return result.reshape(shape)

def linear_srgb_to_acescg(img):
    """Linear sRGB to ACEScg."""
    shape = img.shape
    flat = img.reshape(-1, 3)
    result = flat @ SRGB_TO_ACESCG_MATRIX.T
    return result.reshape(shape)


# =============================================================================
# BUILT-IN COLORSPACE TRANSFORMS
# =============================================================================

BUILTIN_COLORSPACES = [
    "Linear (sRGB primaries)",
    "sRGB",
    "ACEScg",
    "Linear Rec.709",
    "Rec.709",
    "Gamma 2.2",
    "Gamma 2.4", 
    "Raw",
]

def apply_builtin_transform(img_np, src, dst, exposure=0.0):
    """Apply built-in color transform without OCIO."""
    
    result = img_np.copy().astype(np.float32)
    
    # Apply exposure
    if exposure != 0.0:
        result = result * (2.0 ** exposure)
    
    if src == dst:
        return np.clip(result, 0.0, 1.0)
    
    # Normalize colorspace names
    src = src.lower().replace(" ", "").replace("(srgbprimaries)", "")
    dst = dst.lower().replace(" ", "").replace("(srgbprimaries)", "")
    
    # Handle only RGB channels
    if result.shape[-1] >= 3:
        rgb = result[..., :3]
        
        # === Source to Linear ===
        if "srgb" in src and "linear" not in src:
            rgb = srgb_to_linear(rgb)
        elif "rec.709" in src or "rec709" in src:
            if "linear" not in src:
                rgb = rec709_to_linear(rgb)
        elif "acescg" in src:
            rgb = acescg_to_linear_srgb(rgb)  # ACEScg -> Linear sRGB
        elif "gamma2.2" in src:
            rgb = np.power(np.clip(rgb, 0, None), 2.2)
        elif "gamma2.4" in src:
            rgb = np.power(np.clip(rgb, 0, None), 2.4)
        
        # === Linear to Destination ===
        if "srgb" in dst and "linear" not in dst:
            rgb = linear_to_srgb(rgb)
        elif "rec.709" in dst or "rec709" in dst:
            if "linear" not in dst:
                rgb = linear_to_rec709(rgb)
        elif "acescg" in dst:
            rgb = linear_srgb_to_acescg(rgb)  # Linear sRGB -> ACEScg
        elif "gamma2.2" in dst:
            rgb = np.power(np.clip(rgb, 0, None), 1/2.2)
        elif "gamma2.4" in dst:
            rgb = np.power(np.clip(rgb, 0, None), 1/2.4)
        
        result[..., :3] = rgb
    
    return np.clip(result, 0.0, 1.0).astype(np.float32)


# =============================================================================
# OCIO UTILITIES
# =============================================================================

def get_ocio_config():
    """Get OCIO config from environment or return None."""
    if not HAS_OCIO:
        return None
    
    config_path = os.environ.get('OCIO')
    
    if config_path and os.path.exists(config_path):
        try:
            return OCIO.Config.CreateFromFile(config_path)
        except Exception as e:
            print(f"[VFX Bridge] Failed to load OCIO config: {e}")
    
    # Try creating raw config as fallback
    try:
        return OCIO.Config.CreateRaw()
    except:
        return None


def get_ocio_colorspaces(config):
    """Get list of colorspaces from OCIO config."""
    if config is None:
        return BUILTIN_COLORSPACES
    
    try:
        colorspaces = [cs.getName() for cs in config.getColorSpaces()]
        return colorspaces if colorspaces else BUILTIN_COLORSPACES
    except:
        return BUILTIN_COLORSPACES


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
    
    exr_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return exr_files[0]


def get_file_hash(filepath: str) -> str:
    """Get hash of file for change detection."""
    if not os.path.exists(filepath):
        return ""
    stat = os.stat(filepath)
    return f"{stat.st_mtime}_{stat.st_size}"


def load_exr_file(filepath: str) -> tuple[np.ndarray, dict, list[str]]:
    """Load an EXR file and return image data, metadata, and channel list."""
    if not HAS_OPENEXR:
        raise RuntimeError("OpenEXR not installed. Run: pip install OpenEXR Imath")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"EXR file not found: {filepath}")
    
    exr_file = OpenEXR.InputFile(filepath)
    header = exr_file.header()
    
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    channels = list(header['channels'].keys())
    
    first_channel = header['channels'][channels[0]]
    if first_channel.type == Imath.PixelType(Imath.PixelType.HALF):
        pixel_type = Imath.PixelType(Imath.PixelType.HALF)
        np_dtype = np.float16
        bitdepth = 16
    else:
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        np_dtype = np.float32
        bitdepth = 32
    
    channel_data = {}
    for ch_name in channels:
        raw_data = exr_file.channel(ch_name, pixel_type)
        channel_data[ch_name] = np.frombuffer(raw_data, dtype=np_dtype).reshape(height, width)
    
    exr_file.close()
    
    # Detect colorspace from header
    colorspace = "Linear (sRGB primaries)"
    if 'chromaticities' in header:
        chrom = header['chromaticities']
        if hasattr(chrom, 'red') and abs(chrom.red.x - 0.713) < 0.01:
            colorspace = "ACEScg"
    
    metadata = {
        "resolution": (width, height),
        "bitdepth": bitdepth,
        "channels": channels,
        "source_file": filepath,
        "filename": os.path.basename(filepath),
        "colorspace": colorspace,
    }
    
    if 'chromaticities' in header:
        metadata["chromaticities"] = str(header['chromaticities'])
    
    for key in ['framesPerSecond', 'owner', 'comments', 'capDate', 'utcOffset']:
        if key in header:
            metadata[key] = str(header[key])
    
    if 'framesPerSecond' in header:
        fps = header['framesPerSecond']
        metadata['framerate'] = float(fps.n) / float(fps.d) if hasattr(fps, 'n') else float(fps)
    else:
        metadata['framerate'] = None
    
    # Build RGBA image
    if all(ch in channel_data for ch in ['R', 'G', 'B']):
        if 'A' in channel_data:
            image_data = np.stack([
                channel_data['R'], channel_data['G'], 
                channel_data['B'], channel_data['A']
            ], axis=-1).astype(np.float32)
        else:
            image_data = np.stack([
                channel_data['R'], channel_data['G'], channel_data['B']
            ], axis=-1).astype(np.float32)
    else:
        image_data = np.stack(list(channel_data.values()), axis=-1).astype(np.float32)
    
    metadata['_channel_data'] = {k: v.astype(np.float32) for k, v in channel_data.items()}
    
    return image_data, metadata, channels


# =============================================================================
# EXR HOT FOLDER LOADER NODE
# =============================================================================

class EXRHotFolderLoader:
    """Watches a folder and loads the latest EXR file automatically."""
    
    CATEGORY = "VFX Bridge"
    FUNCTION = "load_exr"
    RETURN_TYPES = ("IMAGE", "AOVS", "VFX_METADATA", "STRING", "STRING")
    RETURN_NAMES = ("beauty", "aovs", "metadata", "channels", "filename")
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
                "output_mode": (["raw", "display_ready"], {"default": "raw"}),
            }
        }
    
    @classmethod
    def IS_CHANGED(cls, folder_path, auto_refresh=True):
        if not auto_refresh:
            return False
        latest_exr = get_latest_exr(folder_path)
        if latest_exr:
            return get_file_hash(latest_exr)
        return ""
    
    def load_exr(self, folder_path: str, auto_refresh: bool = True, output_mode: str = "raw"):
        latest_exr = get_latest_exr(folder_path)
        
        if latest_exr is None:
            raise ValueError(f"No EXR files found in: {folder_path}")
        
        image_data, metadata, channels = load_exr_file(latest_exr)
        
        # Output mode: raw = untouched linear data, display_ready = clamped 0-1 for preview
        if output_mode == "display_ready":
            # Simple clamp for preview (use Display Transform node for proper viewing)
            image_data = np.clip(image_data, 0.0, 1.0)
        # raw mode: no processing, keep HDR values as-is
        
        image_tensor = torch.from_numpy(image_data).unsqueeze(0)
        channels_info = ", ".join(channels)
        
        # AOVs: channel data for splitter/selector
        aovs = {
            "channels": metadata.get('_channel_data', {}),
            "resolution": metadata.get('resolution', (0, 0)),
        }
        
        # Clean metadata (no internal data)
        clean_metadata = {k: v for k, v in metadata.items() if not k.startswith('_')}
        
        return (image_tensor, aovs, clean_metadata, channels_info, metadata['filename'])


# =============================================================================
# COLOR TRANSFORM NODE
# =============================================================================

class ColorTransform:
    """
    Apply colorspace transformation.
    Uses OCIO if available, otherwise built-in transforms.
    """
    
    CATEGORY = "VFX Bridge/Color"
    FUNCTION = "transform"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_NODE = False
    
    @classmethod
    def INPUT_TYPES(cls):
        colorspaces = BUILTIN_COLORSPACES
        
        return {
            "required": {
                "image": ("IMAGE",),
                "source_colorspace": (colorspaces, {"default": "Linear (sRGB primaries)"}),
                "target_colorspace": (colorspaces, {"default": "sRGB"}),
            },
            "optional": {
                "exposure": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
            }
        }
    
    def transform(self, image: torch.Tensor, source_colorspace: str, 
                  target_colorspace: str, exposure: float = 0.0):
        
        # Convert to numpy
        if image.dim() == 4:
            img_np = image[0].cpu().numpy()
        else:
            img_np = image.cpu().numpy()
        
        # Apply transform
        result_np = apply_builtin_transform(img_np, source_colorspace, target_colorspace, exposure)
        
        # Convert back to tensor
        result = torch.from_numpy(result_np).unsqueeze(0)
        
        return (result,)


# =============================================================================
# DISPLAY TRANSFORM NODE  
# =============================================================================

class DisplayTransform:
    """
    Apply display transform for preview.
    Converts working colorspace to display-ready output.
    """
    
    CATEGORY = "VFX Bridge/Color"
    FUNCTION = "transform"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)
    OUTPUT_NODE = False
    
    DISPLAYS = ["sRGB Monitor", "Rec.709 TV", "DCI-P3", "Raw"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "input_colorspace": (BUILTIN_COLORSPACES, {"default": "Linear (sRGB primaries)"}),
                "display": (cls.DISPLAYS, {"default": "sRGB Monitor"}),
            },
            "optional": {
                "exposure": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.05}),
            }
        }
    
    def transform(self, image: torch.Tensor, input_colorspace: str, display: str,
                  exposure: float = 0.0, gamma: float = 1.0):
        
        # Convert to numpy
        if image.dim() == 4:
            img_np = image[0].cpu().numpy()
        else:
            img_np = image.cpu().numpy()
        
        # Apply exposure
        if exposure != 0.0:
            img_np = img_np * (2.0 ** exposure)
        
        # Determine target based on display
        if display == "sRGB Monitor":
            target = "sRGB"
        elif display == "Rec.709 TV":
            target = "Rec.709"
        elif display == "DCI-P3":
            target = "Gamma 2.6"  # Approximation
        else:
            target = "Raw"
        
        # Apply transform
        if target != "Raw":
            img_np = apply_builtin_transform(img_np, input_colorspace, target, 0.0)
        
        # Apply gamma
        if gamma != 1.0:
            img_np = np.power(np.clip(img_np, 0.0001, None), 1.0 / gamma)
        
        # Clamp
        img_np = np.clip(img_np, 0.0, 1.0).astype(np.float32)
        
        result = torch.from_numpy(img_np).unsqueeze(0)
        
        return (result,)


# =============================================================================
# COLOR INFO NODE
# =============================================================================


# =============================================================================
# MATTE CHANNEL SPLITTER NODE
# =============================================================================

class MatteChannelSplitter:
    """Splits an EXR into individual matte channels."""
    
    CATEGORY = "VFX Bridge"
    FUNCTION = "split_channels"
    OUTPUT_NODE = False
    MAX_CHANNELS = 16
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aovs": ("AOVS",),
            },
            "optional": {
                "channel_filter": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "e.g., matte, crypto"
                }),
            }
        }
    
    RETURN_TYPES = tuple(["MASK"] * MAX_CHANNELS + ["STRING"])
    RETURN_NAMES = tuple([f"channel_{i}" for i in range(MAX_CHANNELS)] + ["channel_names"])
    
    def split_channels(self, aovs: dict, channel_filter: str = ""):
        if 'channels' not in aovs:
            raise ValueError("No channel data. Connect to EXR Hot Folder Loader.")
        
        channel_data = aovs['channels']
        all_channels = list(channel_data.keys())
        
        if channel_filter.strip():
            filters = [f.strip().lower() for f in channel_filter.split(',')]
            filtered_channels = [ch for ch in all_channels 
                               if any(f in ch.lower() for f in filters)]
        else:
            filtered_channels = all_channels
        
        selected_channels = filtered_channels[:self.MAX_CHANNELS]
        
        outputs = []
        for i in range(self.MAX_CHANNELS):
            if i < len(selected_channels):
                ch_name = selected_channels[i]
                mask_tensor = torch.from_numpy(channel_data[ch_name])
                outputs.append(mask_tensor)
            else:
                if selected_channels:
                    h, w = channel_data[selected_channels[0]].shape
                else:
                    h, w = 512, 512
                outputs.append(torch.zeros(h, w))
        
        channel_names = ", ".join(selected_channels) if selected_channels else "No channels"
        outputs.append(channel_names)
        
        return tuple(outputs)


# =============================================================================
# METADATA DISPLAY NODE
# =============================================================================

class MetadataDisplay:
    """Displays EXR metadata in the ComfyUI interface."""
    
    CATEGORY = "VFX Bridge"
    FUNCTION = "display_metadata"
    RETURN_TYPES = ("STRING", "INT", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("info_text", "width", "height", "bitdepth", "framerate", "colorspace")
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "metadata": ("VFX_METADATA",),
            },
            "optional": {
                "aovs": ("AOVS",),  # backward compatibility
            },
        }
    
    def display_metadata(self, metadata: dict, aovs=None):
        resolution = metadata.get('resolution', (0, 0))
        width, height = resolution
        bitdepth = metadata.get('bitdepth', 16)
        framerate = metadata.get('framerate') or 0.0
        colorspace = metadata.get('colorspace', 'Unknown')
        filename = metadata.get('filename', 'Unknown')
        channels = metadata.get('channels', [])
        
        info_lines = [
            f"File: {filename}",
            f"Resolution: {width} x {height}",
            f"Bit Depth: {bitdepth}-bit",
            f"Framerate: {framerate:.2f} fps" if framerate else "Framerate: N/A",
            f"Colorspace: {colorspace}",
            f"Channels ({len(channels)}): {', '.join(channels[:8])}{'...' if len(channels) > 8 else ''}"
        ]
        info_text = "\n".join(info_lines)
        
        print("\n" + "=" * 40)
        print("VFX Bridge - EXR Metadata")
        print("=" * 40)
        for line in info_lines:
            print(line)
        print("=" * 40 + "\n")
        
        return (info_text, width, height, bitdepth, float(framerate or 0.0), str(colorspace))


# =============================================================================
# EXR SAVE NODE
# =============================================================================

class EXRSaveNode:
    """Saves images back to a 16-bit EXR file with optional color bake and custom AOVs."""
    
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
                    "placeholder": "/path/to/output"
                }),
                "filename": ("STRING", {
                    "default": "output",
                    "multiline": False,
                }),
            },
            "optional": {
                "metadata": ("VFX_METADATA",),
                "aovs": ("AOVS",),  # Original AOVs from EXR loader
                "custom_aov_1": ("MASK",),  # Custom AOV (e.g., depth from ComfyUI)
                "custom_aov_1_name": ("STRING", {"default": ""}),
                "custom_aov_2": ("MASK",),
                "custom_aov_2_name": ("STRING", {"default": ""}),
                "custom_aov_3": ("MASK",),
                "custom_aov_3_name": ("STRING", {"default": ""}),
                "include_original_aovs": ("BOOLEAN", {"default": True}),
                "bitdepth": (["16", "32"], {"default": "16"}),
                "bake_colorspace": ("BOOLEAN", {"default": False}),
                "source_colorspace": (BUILTIN_COLORSPACES, {"default": "Linear (sRGB primaries)"}),
                "output_colorspace": (BUILTIN_COLORSPACES, {"default": "sRGB"}),
            }
        }
    
    def save_exr(self, image: torch.Tensor, output_folder: str, filename: str, 
                 metadata: dict = None, aovs: dict = None,
                 custom_aov_1: torch.Tensor = None, custom_aov_1_name: str = "",
                 custom_aov_2: torch.Tensor = None, custom_aov_2_name: str = "",
                 custom_aov_3: torch.Tensor = None, custom_aov_3_name: str = "",
                 include_original_aovs: bool = True,
                 bitdepth: str = "16", bake_colorspace: bool = False, 
                 source_colorspace: str = "Linear (sRGB primaries)",
                 output_colorspace: str = "sRGB"):
        
        if not HAS_OPENEXR:
            raise RuntimeError("OpenEXR not installed")
        
        os.makedirs(output_folder, exist_ok=True)
        
        # Strip .exr extension for base name
        base_name = filename.replace('.exr', '').rstrip('.')
        
        # Auto-version: find next available version number
        version = 1
        while True:
            if version == 1:
                versioned_name = f"{base_name}.exr"
            else:
                versioned_name = f"{base_name}_v{version:03d}.exr"
            output_path = os.path.join(output_folder, versioned_name)
            if not os.path.exists(output_path):
                break
            version += 1
        
        if image.dim() == 4:
            image_np = image[0].cpu().numpy()
        else:
            image_np = image.cpu().numpy()
        
        # Apply color bake if requested
        if bake_colorspace and source_colorspace != output_colorspace:
            image_np = apply_builtin_transform(image_np, source_colorspace, output_colorspace)
            print(f"[VFX Bridge] Baked: {source_colorspace} -> {output_colorspace}")
        
        height, width = image_np.shape[:2]
        num_channels = image_np.shape[2] if image_np.ndim == 3 else 1
        
        if bitdepth == "16":
            pixel_type = Imath.PixelType(Imath.PixelType.HALF)
            np_dtype = np.float16
        else:
            pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
            np_dtype = np.float32
        
        header = OpenEXR.Header(width, height)
        
        # Start with beauty channels (RGB/RGBA)
        if num_channels >= 3:
            beauty_channels = ['R', 'G', 'B']
            if num_channels >= 4:
                beauty_channels.append('A')
        else:
            beauty_channels = ['Y']
        
        all_channels = {}
        channel_data = {}
        
        # Add beauty channels
        for name in beauty_channels:
            all_channels[name] = Imath.Channel(pixel_type)
        
        for i, name in enumerate(beauty_channels):
            if i < num_channels:
                channel_array = image_np[:, :, i].astype(np_dtype)
            else:
                channel_array = np.zeros((height, width), dtype=np_dtype)
            channel_data[name] = channel_array.tobytes()
        
        # Add original AOVs if provided and requested
        if aovs and include_original_aovs and 'channels' in aovs:
            for aov_name, aov_data in aovs['channels'].items():
                # Skip beauty channels (already added)
                if aov_name in beauty_channels:
                    continue
                all_channels[aov_name] = Imath.Channel(pixel_type)
                # Resize if needed
                aov_array = aov_data
                if aov_array.shape[0] != height or aov_array.shape[1] != width:
                    y_idx = np.linspace(0, aov_array.shape[0] - 1, height).astype(int)
                    x_idx = np.linspace(0, aov_array.shape[1] - 1, width).astype(int)
                    aov_array = aov_array[y_idx][:, x_idx]
                channel_data[aov_name] = aov_array.astype(np_dtype).tobytes()
        
        # Add custom AOVs
        custom_aovs = [
            (custom_aov_1, custom_aov_1_name),
            (custom_aov_2, custom_aov_2_name),
            (custom_aov_3, custom_aov_3_name),
        ]
        
        for custom_mask, custom_name in custom_aovs:
            if custom_mask is not None and custom_name:
                custom_name = custom_name.strip()
                if not custom_name:
                    continue
                # Convert torch tensor to numpy
                if hasattr(custom_mask, 'cpu'):
                    mask_np = custom_mask.cpu().numpy()
                else:
                    mask_np = np.array(custom_mask)
                # Handle batch dimension
                if mask_np.ndim == 3:
                    mask_np = mask_np[0]
                # Resize if needed
                if mask_np.shape[0] != height or mask_np.shape[1] != width:
                    y_idx = np.linspace(0, mask_np.shape[0] - 1, height).astype(int)
                    x_idx = np.linspace(0, mask_np.shape[1] - 1, width).astype(int)
                    mask_np = mask_np[y_idx][:, x_idx]
                all_channels[custom_name] = Imath.Channel(pixel_type)
                channel_data[custom_name] = mask_np.astype(np_dtype).tobytes()
        
        header['channels'] = all_channels
        
        exr_file = OpenEXR.OutputFile(output_path, header)
        exr_file.writePixels(channel_data)
        exr_file.close()
        
        aov_count = len(all_channels) - len(beauty_channels)
        print(f"[VFX Bridge] Saved: {output_path} ({len(beauty_channels)} beauty + {aov_count} AOVs)")
        
        return (output_path,)


# =============================================================================
# UTILITY NODES
# =============================================================================

class PreviewMatte:
    """Preview a matte channel as an image."""
    
    CATEGORY = "VFX Bridge"
    FUNCTION = "preview_matte"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)
    OUTPUT_NODE = False
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"matte": ("MASK",)},
            "optional": {
                "colorize": ("BOOLEAN", {"default": False}),
                "color": (["red", "green", "blue", "white"], {"default": "white"}),
            }
        }
    
    def preview_matte(self, matte: torch.Tensor, colorize: bool = False, color: str = "white"):
        if matte.dim() == 3:
            matte = matte[0]
        
        h, w = matte.shape
        
        if colorize:
            color_map = {"red": [1,0,0], "green": [0,1,0], "blue": [0,0,1], "white": [1,1,1]}
            rgb = color_map.get(color, [1,1,1])
            image = torch.zeros(1, h, w, 3)
            for i, c in enumerate(rgb):
                image[0, :, :, i] = matte * c
        else:
            image = matte.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)
        
        return (image,)


class ChannelSelector:
    """Select a specific channel by name. Connect available_channels to a ShowText node to see options."""
    
    CATEGORY = "VFX Bridge"
    FUNCTION = "select_channel"
    RETURN_TYPES = ("MASK", "STRING", "STRING")
    RETURN_NAMES = ("matte", "channel_name", "available_channels")
    OUTPUT_NODE = False
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aovs": ("AOVS",),
                "channel_name": ("STRING", {"default": "", "multiline": False}),
            },
        }
    
    def select_channel(self, aovs: dict, channel_name: str):
        if 'channels' not in aovs:
            raise ValueError("No channel data in AOVS.")
        
        channel_data = aovs['channels']
        available = list(channel_data.keys())
        available_str = "\n".join(available)
        
        channel_name = channel_name.strip()
        
        # If empty, return first channel and show available
        if not channel_name:
            first_ch = available[0] if available else None
            if first_ch:
                return (torch.from_numpy(channel_data[first_ch]), first_ch, available_str)
            raise ValueError("No channels found in EXR.")
        
        # Exact match
        if channel_name in channel_data:
            return (torch.from_numpy(channel_data[channel_name]), channel_name, available_str)
        
        # Case-insensitive match
        for ch in channel_data.keys():
            if ch.lower() == channel_name.lower():
                return (torch.from_numpy(channel_data[ch]), ch, available_str)
        
        # Partial match (e.g., "matte" matches "matte.R")
        matches = [ch for ch in channel_data.keys() if channel_name.lower() in ch.lower()]
        if matches:
            ch = matches[0]
            return (torch.from_numpy(channel_data[ch]), ch, available_str)
        
        raise ValueError(f"Channel '{channel_name}' not found.\nAvailable:\n{available_str}")


class EXRToImage:
    """Convert EXR float data to standard 0-1 range."""
    
    CATEGORY = "VFX Bridge"
    FUNCTION = "convert"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_NODE = False
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",)},
            "optional": {
                "mode": (["clamp", "normalize", "tonemap"], {"default": "clamp"}),
                "exposure": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
            }
        }
    
    def convert(self, image: torch.Tensor, mode: str = "clamp", exposure: float = 0.0):
        if exposure != 0.0:
            image = image * (2.0 ** exposure)
        
        if mode == "clamp":
            result = torch.clamp(image, 0.0, 1.0)
        elif mode == "normalize":
            img_min, img_max = image.min(), image.max()
            result = (image - img_min) / (img_max - img_min) if img_max > img_min else torch.zeros_like(image)
        elif mode == "tonemap":
            result = torch.clamp(image / (1.0 + image), 0.0, 1.0)
        else:
            result = torch.clamp(image, 0.0, 1.0)
        
        return (result,)


class MaskToImage:
    """Convert MASK to IMAGE."""
    
    CATEGORY = "VFX Bridge"
    FUNCTION = "convert"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_NODE = False
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mask": ("MASK",)}}
    
    def convert(self, mask: torch.Tensor):
        if mask.dim() == 2:
            image = mask.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)
        elif mask.dim() == 3:
            image = mask.unsqueeze(-1).repeat(1, 1, 1, 3)
        else:
            image = mask
        return (torch.clamp(image, 0.0, 1.0),)




class AOVContactSheet:
    """Creates a contact sheet preview of all AOV channels with labels - like Nuke's AOV viewer."""
    
    CATEGORY = "VFX Bridge"
    FUNCTION = "create_contact_sheet"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("contact_sheet", "channel_list")
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aovs": ("AOVS",),
            },
            "optional": {
                "columns": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1}),
                "thumbnail_size": ("INT", {"default": 256, "min": 64, "max": 512, "step": 32}),
                "label_height": ("INT", {"default": 24, "min": 16, "max": 48, "step": 4}),
                "show_beauty": ("BOOLEAN", {"default": True}),
            },
        }
    
    def create_contact_sheet(self, aovs: dict, columns: int = 4, thumbnail_size: int = 256, 
                             label_height: int = 24, show_beauty: bool = True):
        if 'channels' not in aovs:
            raise ValueError("No channel data in AOVS.")
        
        channel_data = aovs['channels']
        channel_names = list(channel_data.keys())
        
        # Optionally add beauty as first item
        items = []
        if show_beauty and 'beauty' in aovs:
            items.append(('beauty (RGB)', aovs['beauty'], True))  # (name, data, is_rgb)
        
        # Add all channels
        for name in channel_names:
            items.append((name, channel_data[name], False))  # (name, data, is_rgb)
        
        if not items:
            raise ValueError("No channels to display.")
        
        num_items = len(items)
        rows = (num_items + columns - 1) // columns
        
        cell_width = thumbnail_size
        cell_height = thumbnail_size + label_height
        sheet_width = columns * cell_width
        sheet_height = rows * cell_height
        
        # Create contact sheet (RGB)
        sheet = np.zeros((sheet_height, sheet_width, 3), dtype=np.float32)
        
        # Fill background with dark gray
        sheet[:, :] = 0.15
        
        for idx, (name, data, is_rgb) in enumerate(items):
            row = idx // columns
            col = idx % columns
            
            x_start = col * cell_width
            y_start = row * cell_height
            
            # Get thumbnail data
            if is_rgb:
                # RGB image - already HWC
                thumb_data = data
            else:
                # Single channel - convert to grayscale RGB
                thumb_data = np.stack([data, data, data], axis=-1)
            
            # Resize to thumbnail size
            h, w = thumb_data.shape[:2]
            if h != thumbnail_size or w != thumbnail_size:
                # Simple resize using numpy
                y_indices = np.linspace(0, h - 1, thumbnail_size).astype(int)
                x_indices = np.linspace(0, w - 1, thumbnail_size).astype(int)
                thumb_data = thumb_data[y_indices][:, x_indices]
            
            # Normalize/clamp for display
            thumb_data = np.clip(thumb_data, 0, 1)
            
            # Place thumbnail
            sheet[y_start:y_start + thumbnail_size, x_start:x_start + thumbnail_size] = thumb_data
            
            # Draw label background
            label_y = y_start + thumbnail_size
            sheet[label_y:label_y + label_height, x_start:x_start + cell_width] = 0.08
            
            # Draw text label (simple pixel text)
            self._draw_text(sheet, name[:20], x_start + 4, label_y + 4, label_height - 8)
        
        # Convert to torch tensor [B, H, W, C]
        contact_sheet = torch.from_numpy(sheet).unsqueeze(0)
        
        channel_list = "\n".join([item[0] for item in items])
        
        return {"ui": {"images": []}, "result": (contact_sheet, channel_list)}
    
    def _draw_text(self, img: np.ndarray, text: str, x: int, y: int, height: int):
        """Draw simple pixel text on the image."""
        # Simple 5x7 pixel font for basic chars
        font = self._get_pixel_font()
        scale = max(1, height // 7)
        
        cursor_x = x
        for char in text:
            if char in font:
                bitmap = font[char]
                for row_idx, row in enumerate(bitmap):
                    for col_idx, pixel in enumerate(row):
                        if pixel:
                            px = cursor_x + col_idx * scale
                            py = y + row_idx * scale
                            for sy in range(scale):
                                for sx in range(scale):
                                    if 0 <= py + sy < img.shape[0] and 0 <= px + sx < img.shape[1]:
                                        img[py + sy, px + sx] = [0.9, 0.9, 0.9]
                cursor_x += (len(bitmap[0]) + 1) * scale
            else:
                cursor_x += 4 * scale
    
    def _get_pixel_font(self):
        """Simple 5-column pixel font."""
        return {
            'A': [[0,1,1,0],[1,0,0,1],[1,1,1,1],[1,0,0,1],[1,0,0,1]],
            'B': [[1,1,1,0],[1,0,0,1],[1,1,1,0],[1,0,0,1],[1,1,1,0]],
            'C': [[0,1,1,1],[1,0,0,0],[1,0,0,0],[1,0,0,0],[0,1,1,1]],
            'D': [[1,1,1,0],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,1,1,0]],
            'E': [[1,1,1,1],[1,0,0,0],[1,1,1,0],[1,0,0,0],[1,1,1,1]],
            'F': [[1,1,1,1],[1,0,0,0],[1,1,1,0],[1,0,0,0],[1,0,0,0]],
            'G': [[0,1,1,1],[1,0,0,0],[1,0,1,1],[1,0,0,1],[0,1,1,0]],
            'H': [[1,0,0,1],[1,0,0,1],[1,1,1,1],[1,0,0,1],[1,0,0,1]],
            'I': [[1,1,1],[0,1,0],[0,1,0],[0,1,0],[1,1,1]],
            'J': [[0,0,1],[0,0,1],[0,0,1],[1,0,1],[0,1,0]],
            'K': [[1,0,0,1],[1,0,1,0],[1,1,0,0],[1,0,1,0],[1,0,0,1]],
            'L': [[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,1,1,1]],
            'M': [[1,0,0,0,1],[1,1,0,1,1],[1,0,1,0,1],[1,0,0,0,1],[1,0,0,0,1]],
            'N': [[1,0,0,1],[1,1,0,1],[1,0,1,1],[1,0,0,1],[1,0,0,1]],
            'O': [[0,1,1,0],[1,0,0,1],[1,0,0,1],[1,0,0,1],[0,1,1,0]],
            'P': [[1,1,1,0],[1,0,0,1],[1,1,1,0],[1,0,0,0],[1,0,0,0]],
            'Q': [[0,1,1,0],[1,0,0,1],[1,0,0,1],[1,0,1,0],[0,1,0,1]],
            'R': [[1,1,1,0],[1,0,0,1],[1,1,1,0],[1,0,1,0],[1,0,0,1]],
            'S': [[0,1,1,1],[1,0,0,0],[0,1,1,0],[0,0,0,1],[1,1,1,0]],
            'T': [[1,1,1,1,1],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0]],
            'U': [[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[0,1,1,0]],
            'V': [[1,0,0,0,1],[1,0,0,0,1],[0,1,0,1,0],[0,1,0,1,0],[0,0,1,0,0]],
            'W': [[1,0,0,0,1],[1,0,0,0,1],[1,0,1,0,1],[1,1,0,1,1],[1,0,0,0,1]],
            'X': [[1,0,0,1],[0,1,1,0],[0,1,1,0],[0,1,1,0],[1,0,0,1]],
            'Y': [[1,0,0,0,1],[0,1,0,1,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0]],
            'Z': [[1,1,1,1],[0,0,1,0],[0,1,0,0],[1,0,0,0],[1,1,1,1]],
            'a': [[0,0,0,0],[0,1,1,0],[1,0,1,1],[1,0,0,1],[0,1,1,1]],
            'b': [[1,0,0,0],[1,1,1,0],[1,0,0,1],[1,0,0,1],[1,1,1,0]],
            'c': [[0,0,0,0],[0,1,1,1],[1,0,0,0],[1,0,0,0],[0,1,1,1]],
            'd': [[0,0,0,1],[0,1,1,1],[1,0,0,1],[1,0,0,1],[0,1,1,1]],
            'e': [[0,1,1,0],[1,0,0,1],[1,1,1,1],[1,0,0,0],[0,1,1,1]],
            'f': [[0,0,1,1],[0,1,0,0],[1,1,1,0],[0,1,0,0],[0,1,0,0]],
            'g': [[0,1,1,1],[1,0,0,1],[0,1,1,1],[0,0,0,1],[0,1,1,0]],
            'h': [[1,0,0,0],[1,1,1,0],[1,0,0,1],[1,0,0,1],[1,0,0,1]],
            'i': [[0,1,0],[0,0,0],[0,1,0],[0,1,0],[0,1,0]],
            'j': [[0,0,1],[0,0,0],[0,0,1],[0,0,1],[0,1,0]],
            'k': [[1,0,0,0],[1,0,1,0],[1,1,0,0],[1,0,1,0],[1,0,0,1]],
            'l': [[1,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,1]],
            'm': [[0,0,0,0,0],[1,1,0,1,0],[1,0,1,0,1],[1,0,1,0,1],[1,0,0,0,1]],
            'n': [[0,0,0,0],[1,1,1,0],[1,0,0,1],[1,0,0,1],[1,0,0,1]],
            'o': [[0,0,0,0],[0,1,1,0],[1,0,0,1],[1,0,0,1],[0,1,1,0]],
            'p': [[0,0,0,0],[1,1,1,0],[1,0,0,1],[1,1,1,0],[1,0,0,0]],
            'q': [[0,0,0,0],[0,1,1,1],[1,0,0,1],[0,1,1,1],[0,0,0,1]],
            'r': [[0,0,0,0],[1,0,1,1],[1,1,0,0],[1,0,0,0],[1,0,0,0]],
            's': [[0,0,0,0],[0,1,1,1],[0,1,0,0],[0,0,1,0],[1,1,1,0]],
            't': [[0,1,0,0],[1,1,1,0],[0,1,0,0],[0,1,0,0],[0,0,1,1]],
            'u': [[0,0,0,0],[1,0,0,1],[1,0,0,1],[1,0,0,1],[0,1,1,1]],
            'v': [[0,0,0,0,0],[1,0,0,0,1],[0,1,0,1,0],[0,1,0,1,0],[0,0,1,0,0]],
            'w': [[0,0,0,0,0],[1,0,0,0,1],[1,0,1,0,1],[1,0,1,0,1],[0,1,0,1,0]],
            'x': [[0,0,0,0],[1,0,0,1],[0,1,1,0],[0,1,1,0],[1,0,0,1]],
            'y': [[0,0,0,0],[1,0,0,1],[0,1,1,1],[0,0,0,1],[0,1,1,0]],
            'z': [[0,0,0,0],[1,1,1,1],[0,0,1,0],[0,1,0,0],[1,1,1,1]],
            '0': [[0,1,1,0],[1,0,0,1],[1,0,0,1],[1,0,0,1],[0,1,1,0]],
            '1': [[0,1,0],[1,1,0],[0,1,0],[0,1,0],[1,1,1]],
            '2': [[0,1,1,0],[1,0,0,1],[0,0,1,0],[0,1,0,0],[1,1,1,1]],
            '3': [[1,1,1,0],[0,0,0,1],[0,1,1,0],[0,0,0,1],[1,1,1,0]],
            '4': [[1,0,0,1],[1,0,0,1],[1,1,1,1],[0,0,0,1],[0,0,0,1]],
            '5': [[1,1,1,1],[1,0,0,0],[1,1,1,0],[0,0,0,1],[1,1,1,0]],
            '6': [[0,1,1,0],[1,0,0,0],[1,1,1,0],[1,0,0,1],[0,1,1,0]],
            '7': [[1,1,1,1],[0,0,0,1],[0,0,1,0],[0,1,0,0],[0,1,0,0]],
            '8': [[0,1,1,0],[1,0,0,1],[0,1,1,0],[1,0,0,1],[0,1,1,0]],
            '9': [[0,1,1,0],[1,0,0,1],[0,1,1,1],[0,0,0,1],[0,1,1,0]],
            '.': [[0],[0],[0],[0],[1]],
            '_': [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,1,1]],
            '-': [[0,0,0,0],[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]],
            ' ': [[0,0],[0,0],[0,0],[0,0],[0,0]],
            '(': [[0,1],[1,0],[1,0],[1,0],[0,1]],
            ')': [[1,0],[0,1],[0,1],[0,1],[1,0]],
        }


# =============================================================================
# NODE MAPPINGS
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "EXRHotFolderLoader": EXRHotFolderLoader,
    "MatteChannelSplitter": MatteChannelSplitter,
    "MetadataDisplay": MetadataDisplay,
    "EXRSaveNode": EXRSaveNode,
    "PreviewMatte": PreviewMatte,
    "ChannelSelector": ChannelSelector,
    "EXRToImage": EXRToImage,
    "MaskToImage": MaskToImage,
    "ColorTransform": ColorTransform,
    "DisplayTransform": DisplayTransform,
    "AOVContactSheet": AOVContactSheet,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EXRHotFolderLoader": "EXR Hot Folder Loader",
    "MatteChannelSplitter": "Matte Channel Splitter", 
    "MetadataDisplay": "Metadata Display",
    "EXRSaveNode": "EXR Save",
    "PreviewMatte": "Preview Matte",
    "ChannelSelector": "Channel Selector",
    "EXRToImage": "EXR to Image",
    "MaskToImage": "Mask to Image",
    "ColorTransform": "Color Transform",
    "DisplayTransform": "Display Transform",
    "AOVContactSheet": "AOV Contact Sheet",
}
