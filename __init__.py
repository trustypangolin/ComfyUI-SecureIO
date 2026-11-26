"""ComfyUI-SecureIO: encrypted outputs and encrypted image inputs for ComfyUI."""

from . import batch, keys, loaders, nodes

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
for _m in (nodes, loaders, keys, batch):
    NODE_CLASS_MAPPINGS.update(_m.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_m.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
