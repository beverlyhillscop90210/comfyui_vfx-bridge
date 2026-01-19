<div align="center">
  <img src="assets/logo.png" alt="VFX Bridge Logo" width="400"/>
  
  # VFX Bridge for ComfyUI
  
  **Seamless VFX Pipeline Integration | 16-bit EXR | AOV Channels | OCIO Color Management**
</div>

---

A custom ComfyUI node package for seamless integration with VFX pipelines (Nuke, Houdini). Load 16-bit EXR files from hotfolders, split matte/AOV channels, preview all passes, preserve metadata, and manage color with OCIO.

## Installation

### Via ComfyUI Manager (Recommended)
1. Open ComfyUI Manager
2. Search for "VFX Bridge"
3. Click Install
4. Restart ComfyUI

### Manual Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/beverlyhillscop90210/comfyui_vfx-bridge
cd comfyui_vfx-bridge
pip install -r requirements.txt
```

---

## Nodes

### EXR Hot Folder Loader

Monitors a folder and automatically loads the most recent EXR file. Designed for live workflows where renders are continuously updated from external applications like Nuke or Houdini.

**Inputs:**
- `folder_path` (STRING): Path to the folder containing EXR files
- `auto_refresh` (BOOLEAN): When enabled, reloads the latest file on each queue

**Outputs:**
- `beauty` (IMAGE): Combined RGB image from the EXR
- `aovs` (AOVS): All channel data for downstream processing
- `metadata` (VFX_METADATA): File information including resolution, bitdepth, colorspace
- `channels` (STRING): Comma-separated list of available channel names
- `filename` (STRING): Name of the loaded file

---

### Matte Channel Splitter

Extracts individual channels from AOVS data and outputs them as separate masks. Provides up to 16 channel outputs simultaneously.

**Inputs:**
- `aovs` (AOVS): Channel data from EXR Hot Folder Loader
- `channel_filter` (STRING, optional): Filter channels by name pattern
- `metadata` (VFX_METADATA): Metadata for reference

**Outputs:**
- `channel_0` through `channel_15` (MASK): Individual channel data
- `channel_names` (STRING): Names of the extracted channels

---

### Channel Selector

Selects a specific channel by name from the AOVS data. Supports partial matching and case-insensitive search.

**Inputs:**
- `aovs` (AOVS): Channel data from EXR Hot Folder Loader
- `channel_name` (STRING): Name of the channel to extract

**Outputs:**
- `matte` (MASK): The selected channel as a mask
- `channel_name` (STRING): Resolved channel name
- `available_channels` (STRING): List of all available channels for reference

---

### AOV Contact Sheet

Creates a grid preview of all AOV channels with labels, similar to the AOV viewer in Nuke. Useful for quickly reviewing all available passes.

**Inputs:**
- `aovs` (AOVS): Channel data from EXR Hot Folder Loader
- `columns` (INT): Number of columns in the grid (1-8)
- `thumbnail_size` (INT): Size of each thumbnail in pixels (64-512)
- `label_height` (INT): Height of the text label area
- `show_beauty` (BOOLEAN): Include the combined RGB image as first item

**Outputs:**
- `contact_sheet` (IMAGE): Grid image showing all channels
- `channel_list` (STRING): List of displayed channel names

---

### Metadata Display

Displays EXR file metadata and outputs individual values for use in other nodes.

**Inputs:**
- `metadata` (VFX_METADATA): Metadata from EXR Hot Folder Loader
- `aovs` (AOVS, optional): For backward compatibility

**Outputs:**
- `info_text` (STRING): Formatted text with all metadata
- `width` (INT): Image width in pixels
- `height` (INT): Image height in pixels
- `bitdepth` (INT): Bit depth (16 or 32)
- `framerate` (FLOAT): Frame rate if available
- `colorspace` (STRING): Embedded colorspace information

---

### Preview Matte

Converts a single-channel MASK to an RGB IMAGE for use with preview nodes. Supports grayscale and false color visualization.

**Inputs:**
- `matte` (MASK): Single channel mask data
- `mode` (COMBO): Display mode - grayscale or false_color

**Outputs:**
- `preview` (IMAGE): RGB image for preview

---

### EXR to Image

Converts HDR float data from EXR files to standard 0-1 range for display and processing. Provides multiple tonemapping options.

**Inputs:**
- `image` (IMAGE): HDR image data
- `mode` (COMBO): Conversion mode - clamp, normalize, or tonemap
- `exposure` (FLOAT): Exposure adjustment in stops (-10 to +10)

**Outputs:**
- `image` (IMAGE): Processed image in 0-1 range

---

### Mask to Image

Simple conversion from MASK to IMAGE format. Converts single-channel data to three-channel grayscale.

**Inputs:**
- `matte` (MASK): Single channel mask

**Outputs:**
- `image` (IMAGE): Grayscale RGB image

---

### Color Transform

Transforms images between colorspaces using built-in matrices. Supports common VFX colorspaces without requiring external OCIO configurations.

**Inputs:**
- `image` (IMAGE): Input image
- `source_colorspace` (COMBO): Current colorspace of the image
- `target_colorspace` (COMBO): Desired output colorspace

**Available Colorspaces:**
- Linear (sRGB primaries)
- sRGB
- ACEScg
- ACES2065-1
- Rec.709
- Gamma 2.2
- Gamma 2.4

**Outputs:**
- `image` (IMAGE): Transformed image

---

### Display Transform

Applies a viewing transform for accurate display of linear or ACEScg footage. Includes exposure and gamma controls.

**Inputs:**
- `image` (IMAGE): Linear or ACEScg image
- `input_colorspace` (COMBO): Source colorspace
- `display` (COMBO): Target display (sRGB Monitor, Rec.709)
- `exposure` (FLOAT): Exposure adjustment in stops
- `gamma` (FLOAT): Gamma adjustment

**Outputs:**
- `preview` (IMAGE): Display-ready image

---

### EXR Save

Exports images back to 16-bit or 32-bit EXR format. Supports including original AOVs and custom passes generated within ComfyUI.

**Inputs:**
- `image` (IMAGE): RGB image to save as beauty pass
- `output_folder` (STRING): Destination folder path
- `filename` (STRING): Output filename (without extension)
- `metadata` (VFX_METADATA, optional): Original metadata for reference
- `aovs` (AOVS, optional): Original AOV data to include
- `custom_aov_1` (MASK, optional): Custom pass to include
- `custom_aov_1_name` (STRING): Channel name for custom pass
- `custom_aov_2` (MASK, optional): Second custom pass
- `custom_aov_2_name` (STRING): Channel name for second pass
- `custom_aov_3` (MASK, optional): Third custom pass
- `custom_aov_3_name` (STRING): Channel name for third pass
- `include_original_aovs` (BOOLEAN): Include all original AOVs in output
- `bitdepth` (COMBO): Output bit depth - 16 or 32
- `bake_colorspace` (BOOLEAN): Apply colorspace transform before saving
- `source_colorspace` (COMBO): Current colorspace of image
- `output_colorspace` (COMBO): Target colorspace for baking

**Outputs:**
- `saved_path` (STRING): Full path to the saved file

---

## Custom Types

| Type | Description |
|------|-------------|
| IMAGE | Standard ComfyUI RGB image tensor |
| MASK | Single channel grayscale tensor |
| AOVS | Dictionary containing all EXR channel data |
| VFX_METADATA | Dictionary containing file metadata |

## Requirements

- ComfyUI 0.8+
- Python 3.10+
- OpenEXR >= 3.2.0
- Imath >= 3.1.0
- NumPy >= 1.24.0
- PyTorch >= 2.0.0

### Optional
- OpenColorIO >= 2.3.0 (for full OCIO configuration support)

## Roadmap

- [x] EXR Hot Folder Loader
- [x] Matte Channel Splitter
- [x] Channel Selector
- [x] AOV Contact Sheet
- [x] Metadata Display
- [x] Color Transform nodes
- [x] EXR Save with custom AOVs
- [ ] Batch Mode (image sequences)
- [ ] Full OCIO configuration file support
- [ ] Cryptomatte decoding

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>Built for the VFX community by peterschings</sub>
</div>
