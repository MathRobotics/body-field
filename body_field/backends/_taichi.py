from __future__ import annotations


def import_taichi():
    try:
        import taichi as ti
    except ImportError as exc:
        raise ImportError(
            "Taichi backends require taichi. Install the `taichi` extra "
            "or use a NumPy backend."
        ) from exc
    return ti


def init_taichi(ti, arch: str) -> None:
    if getattr(ti, "_body_field_initialized", False):
        return
    arch_value = {
        "auto": ti.gpu,
        "gpu": ti.gpu,
        "cpu": ti.cpu,
        "metal": getattr(ti, "metal"),
        "vulkan": getattr(ti, "vulkan"),
        "cuda": getattr(ti, "cuda"),
    }[arch]
    try:
        ti.init(arch=arch_value)
    except Exception:
        if arch in {"auto", "gpu"}:
            ti.init(arch=ti.cpu)
        else:
            raise
    ti._body_field_initialized = True
