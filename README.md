# VFX Bridge for ComfyUI

A custom ComfyUI node package for seamless integration with VFX pipelines (Nuke, Houdini). Load 16-bit EXR files from hotfolders, split matte channels, preserve metadata, and manage color with OCIO.

![VFX Bridge Banner](sample_workflow/banner.png)

## Features

- **EXR Hot Folder Loader** - Automatically loads the latest 16-bit EXR from a watched folder
- **Dynamic Matte Splitter** - Splits all channels/AOVs into separate outputs
- **Metadata Passthrough** - Preserves and displays resolution, framerate, bitdepth, colorspace
- **Non-Destructive Workflow** - Keep original color data, bake only on export
- **OCIO Color Management** *(Phase 3)* - ACES configs, input/output transforms

## Installation

### Via ComfyUI Manager (Recommended)
1. Open ComfyUI Manager
2. Search for "VFX Bridge"
3. Click Install
4. Restart ComfyUI

### Manual Installation
1. Clone this repository into your `ComfyUI/custom_nodes/` directory:
   ```bash
   git clone https://github.com/beverlyhillscop90210/comfyui_vfx-bridge ComfyUI/custom_nodes/comfyui_vfx-bridge
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
   
   For portable version:
   ```bash
   .\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\comfyui_vfx-bridge\requirements.txt
   ```

3. Restart ComfyUI

## Nodes

### 🔥 EXR Hot Folder Loader
Watches a folder and loads the latest EXR file automatically.

| Input | Type | Description |
|-------|------|-------------|
| folder_path | STRING | Path to watch for EXR files |
| auto_refresh | BOOLEAN | Auto-detect new files |

| Output | Type | Description |
|--------|------|-------------|
| image | IMAGE | Loaded EXR as tensor |
| metadata | METADATA | File metadata (resolution, colorspace, etc.) |
| channels | LIST | Available channel names |

### ✂️ Matte Channel Splitter
Splits EXR into individual matte channels.

| Input | Type | Description |
|-------|------|-------------|
| exr_data | EXR_DATA | Loaded EXR data |

| Output | Type | Description |
|--------|------|-------------|
| matte_* | MASK | Individual matte channels (dynamic) |

### 📋 Metadata Display
Shows EXR metadata in the UI.

| Display | Description |
|---------|-------------|
| Resolution | Width × Height |
| Framerate | FPS (if present) |
| Bit Depth | 16-bit / 32-bit |
| Colorspace | ACES / sRGB / etc. |

### 💾 EXR Save Node
Compiles mattes back into a 16-bit EXR with metadata.

| Input | Type | Description |
|-------|------|-------------|
| mattes | MASK[] | Matte channels to compile |
| metadata | METADATA | Original metadata to preserve |
| output_path | STRING | Save location |
| bake_colorspace | BOOLEAN | Bake OCIO transform on export |

## Usage

1. Add **EXR Hot Folder Loader** node
2. Set the folder path to your Nuke/Houdini render output
3. Connect to **Matte Channel Splitter**
4. Use individual matte outputs in your ComfyUI pipeline
5. Connect **Metadata Display** to see file info
6. Use **EXR Save Node** to compile back to EXR

## Requirements

- ComfyUI
- Python 3.10+
- OpenEXR >= 3.2.0
- NumPy >= 1.24.0

## Roadmap

- [x] Project setup
- [ ] EXR Hot Folder Loader
- [ ] Matte Channel Splitter
- [ ] Metadata Display
- [ ] EXR Save Node
- [ ] Batch Mode (image sequences)
- [ ] OCIO Color Management

## Credits

Built for the VFX community by [beverlyhillscop90210](https://github.com/beverlyhillscop90210).

Inspired by Nuke and Houdini workflows.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=beverlyhillscop90210/comfyui_vfx-bridge&type=Date)](https://star-history.com/#beverlyhillscop90210/comfyui_vfx-bridge&Date)
