"""
ComfyUI VFX Bridge
A custom node package for EXR hotfolder loading, matte splitting, 
and OCIO color management for Nuke/Houdini integration.

by peterschings
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Web directory for custom JavaScript styling
WEB_DIRECTORY = "./web/js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

__version__ = "0.0.1"
