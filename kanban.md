# ComfyUI VFX Bridge - Project Kanban

> **Project**: comfyui_vfx-bridge  
> **Goal**: EXR hotfolder loader with matte splitting, metadata passthrough, and OCIO color management for Nuke/Houdini integration  
> **Repository**: https://github.com/beverlyhillscop90210/comfyui_vfx-bridge

---

## 📋 Backlog

### Phase 2 - Batch Mode
- [ ] Batch loader for image sequences (frame range input)
- [ ] Frame counter / progress tracking
- [ ] Memory-efficient batch processing

### Phase 3 - Color Management (OCIO)
- [ ] Research Nuke's OCIO implementation & best practices
- [ ] OCIO config file loader (fetch ACES configs)
- [ ] Input colorspace dropdown (dynamic from OCIO config)
- [ ] Output/Viewport colorspace dropdown
- [ ] View Transform node (preview only, non-destructive)
- [ ] Export bake option (apply colorspace on save)

### Future Ideas
- [ ] Multi-layer EXR write node (compile mattes back)
- [ ] Timecode / frame metadata handling
- [ ] Integration with ComfyUI workflow templates

---

## 🚧 In Progress

*(Nothing yet)*

---

## ✅ Done

### 1. Project Setup
- [x] Initialize project structure (`__init__.py`, `nodes.py`, etc.)
- [x] Create `pyproject.toml` for registry publishing
- [x] Create `requirements.txt` (OpenEXR, numpy, etc.)
- [x] Setup GitHub Actions for publishing
- [x] Add LICENSE file (MIT)
- [x] Add README.md with usage instructions
- [x] Initialize git repository

---

## 🎯 Phase 1 - Core Single Still (Current Focus)

### 2. EXR Hot Folder Loader Node
- [ ] Folder path input (user selects hotfolder)
- [ ] Detect latest EXR file in folder (by modification time)
- [ ] Load 16-bit EXR file
- [ ] Dynamic channel detection (list all available channels)
- [ ] IS_CHANGED implementation (detect new files)

### 3. Matte Channel Splitter Node
- [ ] Input: loaded EXR data
- [ ] Dynamic output generation per channel
- [ ] Convert channels to ComfyUI IMAGE tensors
- [ ] Preview capability for each channel

### 4. Metadata Display
- [ ] Extract and display:
  - Resolution (width x height)
  - Framerate (if present)
  - Bit depth
  - Color profile / colorspace
- [ ] Pass metadata as linkable output

### 5. Metadata Passthrough
- [ ] Create METADATA datatype
- [ ] Link metadata from loader → exporter
- [ ] Preserve original metadata through pipeline

### 6. EXR Save Node (Basic)
- [ ] Accept matte channels as inputs
- [ ] Accept metadata input
- [ ] Compile back to 16-bit EXR
- [ ] Output path selection

---

## 📝 Notes

### Technical Stack
- **OpenEXR**: Python bindings for EXR read/write
- **OpenImageIO** (optional): Robust EXR handling
- **NumPy**: Array operations
- **PyTorch**: Tensor conversion (already in ComfyUI)
- **OpenColorIO**: Color management (Phase 3)

### Key Decisions
1. **Dynamic channels**: Auto-detect all EXR layers/channels
2. **Non-destructive color**: Keep original colorspace, bake only on export
3. **OCIO approach**: Follow Nuke's patterns, load ACES configs dynamically
4. **Metadata as data**: Treat metadata as a passable datatype

### Metadata Schema (Draft)
```python
{
    "resolution": (width, height),
    "framerate": float or None,
    "bitdepth": 16,  # or 32
    "colorspace": "ACES - ACEScg",  # or detected value
    "channels": ["R", "G", "B", "A", "matte1.R", ...],
    "source_file": "/path/to/file.exr",
    "custom": {}  # Any additional Nuke/Houdini attributes
}
```

---

## 🏷️ Version History

| Version | Date       | Notes |
|---------|------------|-------|
| 0.1.0   | 2026-01-18 | Initial project setup |

