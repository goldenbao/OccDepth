import os
import sys
import types

import torch
import torch.nn as nn


IGEV_RR_CODE_DIR = (
    "/home/data/bino_stereo/binocularstereovision"
    "/scratch_igev_rr_except_mn2_on_sceneflow_w_aug/code"
)


def _import_igev_rr():
    """Import IGEVRRStereo from the IGEV-RR project, bypassing problematic
    __init__.py imports (casnet/h5py etc.)."""
    if "stereo.modeling.models.igev_rr.igev_rr_stereo" in sys.modules:
        return sys.modules["stereo.modeling.models.igev_rr.igev_rr_stereo"]

    code_dir = IGEV_RR_CODE_DIR
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)

    # Build the package hierarchy manually so that relative imports
    # (from .update, from .extractor, etc.) resolve correctly.
    def _ensure_package(name, path):
        if name in sys.modules:
            return sys.modules[name]
        pkg = types.ModuleType(name)
        pkg.__path__ = [path]
        pkg.__file__ = os.path.join(path, "__init__.py") if os.path.isdir(path) else path
        sys.modules[name] = pkg
        return pkg

    # create empty packages for the IGEV-RR code structure
    _ensure_package("stereo", os.path.join(code_dir, "stereo"))
    _ensure_package("stereo.modeling", os.path.join(code_dir, "stereo", "modeling"))
    _ensure_package(
        "stereo.modeling.models",
        os.path.join(code_dir, "stereo", "modeling", "models"),
    )
    # igev_rr subpackage (no __init__.py exists, create a dummy one)
    igev_rr_pkg = types.ModuleType("stereo.modeling.models.igev_rr")
    igev_rr_pkg.__path__ = [
        os.path.join(code_dir, "stereo", "modeling", "models", "igev_rr")
    ]
    sys.modules["stereo.modeling.models.igev_rr"] = igev_rr_pkg

    import stereo.modeling.models.igev_rr.igev_rr_stereo as _m
    return _m


class IGEVRRWrapper(nn.Module):
    """Frozen IGEV-RR stereo matching model wrapped as an OccDepth submodule.

    Takes left/right image pairs (in ``[0, 255]`` float32 range) and returns
    disparity maps and depth maps.
    """

    def __init__(
        self,
        ckpt_path,
        max_disp=192,
        hidden_dims=None,
        n_gru_iters=2,
        mobilenetv2_075=False,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128, 128]

        _mod = _import_igev_rr()
        args = _make_igev_args(max_disp, hidden_dims, n_gru_iters, mobilenetv2_075)
        self.model = _mod.IGEVRRStereo(args)

        # Load checkpoint weights — fail loudly if missing
        if not ckpt_path:
            raise FileNotFoundError(
                f"IGEVRRWrapper: igev_rr_ckpt is empty. "
                f"Set igev_rr_ckpt in your config yaml."
            )
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"IGEVRRWrapper: checkpoint not found at:\n  {ckpt_path}\n"
                f"Verify igev_rr_ckpt in your config yaml points to a valid .pth file."
            )
        _load_ckpt(self.model, ckpt_path)

        # Freeze everything
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, left_img, right_img):
        """Forward pass.

        Args:
            left_img:  (B, 3, H, W) float32 in [0, 255]
            right_img: (B, 3, H, W) float32 in [0, 255]

        Returns:
            dict with keys:
                disp_pred  (B, H, W)  disparity in pixels
        """
        data = {"left": left_img, "right": right_img}
        with torch.no_grad():
            out = self.model(data)
        return out


def _make_igev_args(max_disp, hidden_dims, n_gru_iters, mobilenetv2_075):
    class _Args:
        pass
    a = _Args()
    a.MAX_DISP = max_disp
    a.MAX_DISP_SCALE = None
    a.HIDDEN_DIMS = hidden_dims
    a.N_GRU_ITERS = n_gru_iters
    a.MOBILENETV2_075 = mobilenetv2_075
    a.BIDIRECTIONAL = False
    a.DISTILL = False
    a.MIN_DISP = 0
    a.RESIZE_SCALE = 1.0
    a.CORR_LEVELS = 2
    a.CORR_RADIUS = 4
    a.SLOW_FAST_GRU = True
    a.N_DOWNSAMPLE = 2
    return a


def _load_ckpt(model, ckpt_path):
    raw = torch.load(ckpt_path, map_location="cpu")
    state_dict = raw.get("model_state", raw)

    model_sd = model.state_dict()
    update = {}
    for key, val in state_dict.items():
        # Handle possible DataParallel prefix
        clean = key.replace("module.", "")
        if clean in model_sd and model_sd[clean].shape == val.shape:
            update[clean] = val
    model_sd.update(update)
    model.load_state_dict(model_sd)
    n_loaded = len(update)
    total = len(model_sd)
    if n_loaded != total:
        missing = set(model_sd.keys()) - set(update.keys())
        print(f"IGEVRRWrapper: loaded {n_loaded}/{total} params from {ckpt_path}")
        print(f"  Missing keys ({len(missing)}):")
        for k in sorted(missing)[:10]:
            print(f"    {k}")
        raise RuntimeError(
            f"IGEVRRWrapper: only {n_loaded}/{total} parameters loaded from {ckpt_path}. "
            f"Checkpoint key mismatch — check whether the .pth was trained with "
            f"a different architecture (DataParallel wrapper, different backbone, etc.)."
        )
    print(f"IGEVRRWrapper: loaded {n_loaded}/{total} params from {ckpt_path}")
