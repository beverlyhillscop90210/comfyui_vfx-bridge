"""
ComfyUI VFX Bridge
A custom node package for EXR hotfolder loading, matte splitting, 
and OCIO color management for Nuke/Houdini integration.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

__version__ = "0.1.0"
