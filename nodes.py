"""
ComfyUI VFX Bridge - Node Definitions
Full OCIO (OpenColorIO) Integration for VFX Pipelines
"""

import os
import glob
import hashlib
import json
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
    print(f"[VFX Bridge] OCIO {OCIO.__version__} loaded successfully")
except ImportError:
    HAS_OCIO = False
    print("[VFX Bridge] Warning: OpenColorIO not installed. Run: pip install opencolorio")


# =============================================================================
# OCIO UTILITIES
# =============================================================================

def get_ocio_config():
    """Get the current OCIO config from environment or default."""
    if not HAS_OCIO:
        return None
    
    # Try environment variable first
    config_path = os.environ.get('OCIO')
    
    if config_path and os.path.exists(config_path):
        try:
            return OCIO.Config.CreateFromFile(config_path)
        except Exception as e:
            print(f"[VFX Bridge] Failed to load OCIO config from {config_path}: {e}")
    
    # Fall back to built-in config
    try:
        return OCIO.Config.CreateFromBuiltinConfig("aces_1.2")
    except:
        try:
            return OCIO.Config.CreateRaw()
        except:
            return None


def get_ocio_colorspaces(config):
    """Get list of available colorspaces from OCIO config."""
    if config is None:
        return ["sRGB", "Linear", "ACEScg", "ACES2065-1", "Raw"]
    
    colorspaces = []
    for i in range(config.getNumColorSpaces()):
        colorspaces.append(config.getColorSpaceNameByIndex(i))
    
    return colorspaces if colorspaces else ["sRGB", "Linear", "ACEScg", "Raw"]


def get_ocio_displays(config):
    """Get list of available displays from OCIO config."""
    if config is None:
        return ["sRGB"]
    
    displays = []
    for i in range(config.getNumDisplays()):
        displays.append(config.getDisplay(i))
    
    return displays if displays else ["sRGB"]


def get_ocio_views(config, display):
    """Get list of available views for a display."""
    if config is None:
        return ["Standard"]
    
    views = []
    try:
        for i in range(config.getNumViews(display)):
            views.append(config.getView(display, i))
    except:
        views = ["Standard"]
    
    return views if views else ["Standard"]


def apply_ocio_transform(image_np, src_colorspace, dst_colorspace, config=None):
    """Apply OCIO colorspace transform to numpy image."""
    if not HAS_OCIO:
        print("[VFX Bridge] OCIO not available, returning unchanged")
        return image_np
    
    if config is None:
        config = get_ocio_config()
    
    if config is None:
        return image_np
    
    try:
        processor = config.getProcessor(src_colorspace, dst_colorspace)
        cpu = processor.getDefaultCPUProcessor()
        
        # OCIO expects float32
        img = image_np.astype(np.float32)
        
        # Process the image
        if img.ndim == 3:
            # [H, W, C] - process directly
            cpu.applyRGBA(img) if img.shape[2] == 4 else cpu.applyRGB(img)
        
        return img
        
    except Exception as e:
        print(f"[VFX Bridge] OCIO transform failed: {e}")
        return image_np


def apply_ocio_display_transform(image_np, src_colorspace, display, view, config=None):
    """Apply OCIO display transform to numpy image."""
    if not HAS_OCIO:
        return image_np
    
    if config is None:
        config = get_ocio_config()
    
    if config is None:
        return image_np
    
    try:
        # Create display transform
        transform = OCIO.DisplayViewTransform()
        transform.setSrc(src_colorspace)
        transform.setDisplay(display)
        transform.setView(view)
        
        processor = config.getProcessor(transform)
        cpu = processor.getDefaultCPUProcessor()
        
        # Process image
        img = image_np.astype(np.float32).copy()
        
        if img.ndim == 3:
            h, w, c = img.shape
            # Flatten for OCIO
            pixels = img.reshape(-1, c)
            
            if c >= 3:
                # Process RGB(A)
                for i in range(len(pixels)):
                    if c == 4:
                        pixels[i] = cpu.applyRGBA(pixels[i])
                    else:
                        pixels[i, :3] = cpu.applyRGB(pixels[i, :3])
            
            img = pixels.reshape(h, w, c)
        
        return img
        
    except Exception as e:
        print(f"[VFX Bridge] OCIO display transform failed: {e}")
        return image_np


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
    colorspace = "Linear"  # Default assumption for EXR
    if 'chromaticities' in header:
        chrom = header['chromaticities']
        # Check for ACES primaries
        if hasattr(chrom, 'red') and abs(chrom.red.x - 0.64) < 0.01:
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
        if not auto_refresh:
            return False
        latest_exr = get_latest_exr(folder_path)
        if latest_exr:
            return get_file_hash(latest_exr)
        return ""
    
    def load_exr(self, folder_path: str, auto_refresh: bool = True):
        latest_exr = get_latest_exr(folder_path)
        
        if latest_exr is None:
            raise ValueError(f"No EXR files found in: {folder_path}")
        
        image_data, metadata, channels = load_exr_file(latest_exr)
        image_tensor = torch.from_numpy(image_data).unsqueeze(0)
        channels_info = ", ".join(channels)
        
        return (image_tensor, metadata, channels_info, metadata['filename'])


# =============================================================================
# OCIO COLOR TRANSFORM NODE
# =============================================================================

class OCIOColorTransform:
    """
    Apply OCIO colorspace transformation.
    Non-destructive: use for preview or bake on export.
    """
    
    CATEGORY = "VFX Bridge/OCIO"
    FUNCTION = "transform"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_NODE = False
    
    # Common colorspaces (will be extended by OCIO config at runtime)
    COLORSPACES = [
        "Linear", "sRGB", "ACEScg", "ACES2065-1", 
        "Linear Rec.709", "Linear P3-D65",
        "Gamma 2.2", "Gamma 2.4",
        "Raw"
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        config = get_ocio_config()
        colorspaces = get_ocio_colorspaces(config) if config else cls.COLORSPACES
        
        return {
            "required": {
                "image": ("IMAGE",),
                "source_colorspace": (colorspaces, {"default": colorspaces[0] if colorspaces else "Linear"}),
                "target_colorspace": (colorspaces, {"default": "sRGB" if "sRGB" in colorspaces else colorspaces[0]}),
            },
            "optional": {
                "exposure": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
            }
        }
    
    def transform(self, image: torch.Tensor, source_colorspace: str, 
                  target_colorspace: str, exposure: float = 0.0):
        
        # Apply exposure
        if exposure != 0.0:
            image = image * (2.0 ** exposure)
        
        # Convert to numpy for OCIO
        if image.dim() == 4:
            img_np = image[0].cpu().numpy()
        else:
            img_np = image.cpu().numpy()
        
        # Apply OCIO transform
        if HAS_OCIO and source_colorspace != target_colorspace:
            config = get_ocio_config()
            if config:
                try:
                    processor = config.getProcessor(source_colorspace, target_colorspace)
                    cpu = processor.getDefaultCPUProcessor()
                    
                    h, w, c = img_np.shape
                    img_flat = img_np.reshape(-1, c).copy()
                    
                    # Apply transform pixel by pixel (safe method)
                    for i in range(len(img_flat)):
                        if c >= 4:
                            img_flat[i] = cpu.applyRGBA(img_flat[i])
                        elif c == 3:
                            img_flat[i] = cpu.applyRGB(img_flat[i])
                    
                    img_np = img_flat.reshape(h, w, c)
                    
                except Exception as e:
                    print(f"[VFX Bridge] OCIO transform error: {e}")
        
        # Convert back to tensor
        result = torch.from_numpy(img_np).unsqueeze(0)
        
        return (result,)


# =============================================================================
# OCIO DISPLAY TRANSFORM NODE  
# =============================================================================

class OCIODisplayTransform:
    """
    Apply OCIO display/view transform for preview.
    Converts working colorspace to display colorspace.
    """
    
    CATEGORY = "VFX Bridge/OCIO"
    FUNCTION = "transform"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)
    OUTPUT_NODE = False
    
    @classmethod
    def INPUT_TYPES(cls):
        config = get_ocio_config()
        colorspaces = get_ocio_colorspaces(config) if config else ["Linear", "ACEScg", "sRGB"]
        displays = get_ocio_displays(config) if config else ["sRGB", "Rec.709"]
        
        return {
            "required": {
                "image": ("IMAGE",),
                "input_colorspace": (colorspaces, {"default": "ACEScg" if "ACEScg" in colorspaces else colorspaces[0]}),
                "display": (displays, {"default": displays[0]}),
            },
            "optional": {
                "exposure": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.05}),
            }
        }
    
    def transform(self, image: torch.Tensor, input_colorspace: str, display: str,
                  exposure: float = 0.0, gamma: float = 1.0):
        
        # Apply exposure
        result = image.clone()
        if exposure != 0.0:
            result = result * (2.0 ** exposure)
        
        # Convert to numpy
        if result.dim() == 4:
            img_np = result[0].cpu().numpy()
        else:
            img_np = result.cpu().numpy()
        
        # Apply OCIO display transform
        if HAS_OCIO:
            config = get_ocio_config()
            if config:
                try:
                    views = get_ocio_views(config, display)
                    view = views[0] if views else "Standard"
                    
                    transform = OCIO.DisplayViewTransform()
                    transform.setSrc(input_colorspace)
                    transform.setDisplay(display)
                    transform.setView(view)
                    
                    processor = config.getProcessor(transform)
                    cpu = processor.getDefaultCPUProcessor()
                    
                    h, w, c = img_np.shape
                    img_flat = img_np.reshape(-1, c).copy()
                    
                    for i in range(len(img_flat)):
                        if c >= 4:
                            img_flat[i] = cpu.applyRGBA(img_flat[i])
                        elif c == 3:
                            img_flat[i] = cpu.applyRGB(img_flat[i])
                    
                    img_np = img_flat.reshape(h, w, c)
                    
                except Exception as e:
                    print(f"[VFX Bridge] OCIO display transform error: {e}, using fallback")
                    # Fallback to simple sRGB
                    img_np = np.where(img_np <= 0.0031308,
                                      img_np * 12.92,
                                      1.055 * np.power(np.clip(img_np, 0.0031308, None), 1/2.4) - 0.055)
        else:
            # Fallback without OCIO: simple linear to sRGB
            img_np = np.where(img_np <= 0.0031308,
                              img_np * 12.92,
                              1.055 * np.power(np.clip(img_np, 0.0031308, None), 1/2.4) - 0.055)
        
        # Apply gamma
        if gamma != 1.0:
            img_np = np.power(np.clip(img_np, 0.0001, None), 1.0 / gamma)
        
        # Clamp and convert back
        img_np = np.clip(img_np, 0.0, 1.0)
        result = torch.from_numpy(img_np.astype(np.float32)).unsqueeze(0)
        
        return (result,)


# =============================================================================
# OCIO CONFIG INFO NODE
# =============================================================================

class OCIOConfigInfo:
    """
    Display information about the current OCIO configuration.
    Shows available colorspaces, displays, and views.
    """
    
    CATEGORY = "VFX Bridge/OCIO"
    FUNCTION = "get_info"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("config_path", "colorspaces", "displays", "ocio_version")
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "custom_config": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to custom OCIO config (optional)"
                }),
            }
        }
    
    def get_info(self, custom_config: str = ""):
        if not HAS_OCIO:
            return ("OCIO not installed", "N/A", "N/A", "N/A")
        
        # Get config
        if custom_config and os.path.exists(custom_config):
            try:
                config = OCIO.Config.CreateFromFile(custom_config)
                config_path = custom_config
            except:
                config = get_ocio_config()
                config_path = os.environ.get('OCIO', 'Built-in ACES 1.2')
        else:
            config = get_ocio_config()
            config_path = os.environ.get('OCIO', 'Built-in ACES 1.2')
        
        if config is None:
            return (config_path, "No config loaded", "N/A", OCIO.__version__)
        
        # Get colorspaces
        colorspaces = get_ocio_colorspaces(config)
        colorspaces_str = ", ".join(colorspaces[:20])
        if len(colorspaces) > 20:
            colorspaces_str += f"... (+{len(colorspaces) - 20} more)"
        
        # Get displays
        displays = get_ocio_displays(config)
        displays_str = ", ".join(displays)
        
        # Print info
        print("\n" + "=" * 50)
        print("VFX Bridge - OCIO Configuration")
        print("=" * 50)
        print(f"Config: {config_path}")
        print(f"Version: {OCIO.__version__}")
        print(f"Colorspaces: {len(colorspaces)}")
        print(f"Displays: {displays_str}")
        print("=" * 50 + "\n")
        
        return (config_path, colorspaces_str, displays_str, OCIO.__version__)


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
    
    RETURN_TYPES = tuple(["MASK"] * MAX_CHANNELS + ["STRING"])
    RETURN_NAMES = tuple([f"channel_{i}" for i in range(MAX_CHANNELS)] + ["channel_names"])
    
    def split_channels(self, metadata: dict, channel_filter: str = ""):
        if '_channel_data' not in metadata:
            raise ValueError("No channel data in metadata. Connect to EXR Hot Folder Loader.")
        
        channel_data = metadata['_channel_data']
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
        }
    
    def display_metadata(self, metadata: dict):
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
        
        print("\n" + "=" * 50)
        print("VFX Bridge - EXR Metadata")
        print("=" * 50)
        for line in info_lines:
            print(line)
        print("=" * 50 + "\n")
        
        return (info_text, width, height, bitdepth, float(framerate or 0.0), str(colorspace))


# =============================================================================
# EXR SAVE NODE (WITH OCIO BAKE OPTION)
# =============================================================================

class EXRSaveNode:
    """
    Saves images/mattes back to a 16-bit EXR file.
    Option to bake OCIO colorspace on export.
    """
    
    CATEGORY = "VFX Bridge"
    FUNCTION = "save_exr"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        config = get_ocio_config()
        colorspaces = get_ocio_colorspaces(config) if config else ["Linear", "sRGB", "ACEScg"]
        
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
                "bake_colorspace": ("BOOLEAN", {"default": False}),
                "source_colorspace": (colorspaces, {"default": "Linear"}),
                "output_colorspace": (colorspaces, {"default": "sRGB" if "sRGB" in colorspaces else colorspaces[0]}),
            }
        }
    
    def save_exr(self, image: torch.Tensor, output_folder: str, filename: str, 
                 metadata: dict = None, bitdepth: str = "16",
                 bake_colorspace: bool = False, source_colorspace: str = "Linear",
                 output_colorspace: str = "sRGB"):
        
        if not HAS_OPENEXR:
            raise RuntimeError("OpenEXR not installed. Run: pip install OpenEXR Imath")
        
        os.makedirs(output_folder, exist_ok=True)
        
        if not filename.endswith('.exr'):
            filename = f"{filename}.exr"
        output_path = os.path.join(output_folder, filename)
        
        if image.dim() == 4:
            image_np = image[0].cpu().numpy()
        else:
            image_np = image.cpu().numpy()
        
        # Apply OCIO bake if requested
        if bake_colorspace and HAS_OCIO and source_colorspace != output_colorspace:
            config = get_ocio_config()
            if config:
                try:
                    processor = config.getProcessor(source_colorspace, output_colorspace)
                    cpu = processor.getDefaultCPUProcessor()
                    
                    h, w, c = image_np.shape
                    img_flat = image_np.reshape(-1, c).copy()
                    
                    for i in range(len(img_flat)):
                        if c >= 4:
                            img_flat[i] = cpu.applyRGBA(img_flat[i])
                        elif c == 3:
                            img_flat[i] = cpu.applyRGB(img_flat[i])
                    
                    image_np = img_flat.reshape(h, w, c)
                    print(f"[VFX Bridge] Baked colorspace: {source_colorspace} -> {output_colorspace}")
                    
                except Exception as e:
                    print(f"[VFX Bridge] OCIO bake failed: {e}")
        
        height, width = image_np.shape[:2]
        num_channels = image_np.shape[2] if image_np.ndim == 3 else 1
        
        if bitdepth == "16":
            pixel_type = Imath.PixelType(Imath.PixelType.HALF)
            np_dtype = np.float16
        else:
            pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
            np_dtype = np.float32
        
        header = OpenEXR.Header(width, height)
        
        if num_channels >= 3:
            channel_names = ['R', 'G', 'B']
            if num_channels >= 4:
                channel_names.append('A')
        else:
            channel_names = ['Y']
        
        header['channels'] = {name: Imath.Channel(pixel_type) for name in channel_names}
        
        channel_data = {}
        for i, name in enumerate(channel_names):
            if i < num_channels:
                channel_array = image_np[:, :, i].astype(np_dtype)
            else:
                channel_array = np.zeros((height, width), dtype=np_dtype)
            channel_data[name] = channel_array.tobytes()
        
        exr_file = OpenEXR.OutputFile(output_path, header)
        exr_file.writePixels(channel_data)
        exr_file.close()
        
        print(f"[VFX Bridge] Saved EXR: {output_path}")
        
        return (output_path,)


# =============================================================================
# PREVIEW MATTE NODE
# =============================================================================

class PreviewMatte:
    """Preview a single matte channel as an image."""
    
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
        if matte.dim() == 3:
            matte = matte[0]
        
        h, w = matte.shape
        
        if colorize:
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
            image = matte.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)
        
        return (image,)


# =============================================================================
# CHANNEL SELECTOR NODE
# =============================================================================

class ChannelSelector:
    """Select a specific channel from EXR by name."""
    
    CATEGORY = "VFX Bridge"
    FUNCTION = "select_channel"
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("matte", "channel_name")
    OUTPUT_NODE = False
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "metadata": ("VFX_METADATA",),
                "channel_name": ("STRING", {
                    "default": "R",
                    "multiline": False,
                    "placeholder": "R, G, B, A, matte.R, etc."
                }),
            },
        }
    
    def select_channel(self, metadata: dict, channel_name: str):
        if '_channel_data' not in metadata:
            raise ValueError("No channel data in metadata.")
        
        channel_data = metadata['_channel_data']
        available = list(channel_data.keys())
        channel_name = channel_name.strip()
        
        if channel_name not in channel_data:
            for ch in available:
                if ch.lower() == channel_name.lower():
                    channel_name = ch
                    break
            else:
                raise ValueError(f"Channel '{channel_name}' not found. Available: {', '.join(available)}")
        
        mask_tensor = torch.from_numpy(channel_data[channel_name])
        
        return (mask_tensor, channel_name)


# =============================================================================
# EXR TO IMAGE NODE
# =============================================================================

class EXRToImage:
    """Convert EXR float data to standard ComfyUI IMAGE format."""
    
    CATEGORY = "VFX Bridge"
    FUNCTION = "convert"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_NODE = False
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
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
            img_min = image.min()
            img_max = image.max()
            if img_max > img_min:
                result = (image - img_min) / (img_max - img_min)
            else:
                result = torch.zeros_like(image)
        elif mode == "tonemap":
            result = image / (1.0 + image)
            result = torch.clamp(result, 0.0, 1.0)
        else:
            result = torch.clamp(image, 0.0, 1.0)
        
        return (result,)


# =============================================================================
# MASK TO IMAGE NODE
# =============================================================================

class MaskToImage:
    """Convert a MASK to a standard IMAGE."""
    
    CATEGORY = "VFX Bridge"
    FUNCTION = "convert"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_NODE = False
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            },
        }
    
    def convert(self, mask: torch.Tensor):
        if mask.dim() == 2:
            image = mask.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)
        elif mask.dim() == 3:
            image = mask.unsqueeze(-1).repeat(1, 1, 1, 3)
        else:
            image = mask
        
        image = torch.clamp(image, 0.0, 1.0)
        
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
    "ChannelSelector": ChannelSelector,
    "EXRToImage": EXRToImage,
    "MaskToImage": MaskToImage,
    "OCIOColorTransform": OCIOColorTransform,
    "OCIODisplayTransform": OCIODisplayTransform,
    "OCIOConfigInfo": OCIOConfigInfo,
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
    "OCIOColorTransform": "OCIO Color Transform",
    "OCIODisplayTransform": "OCIO Display Transform",
    "OCIOConfigInfo": "OCIO Config Info",
}
