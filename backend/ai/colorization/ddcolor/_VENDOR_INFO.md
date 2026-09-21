# Vendored DDColor sources

- Upstream: https://github.com/piddnad/DDColor
- Revision: `2adb63f2656ac41cbdf7b894cddd94121a3faf13`
- License:  Apache-2.0 (DDColor — Alibaba DAMO Academy)
- Fetched by: `backend/ai/install_ddcolor.py`

## Files

- `model.py` ← `ddcolor/model.py`
- `pipeline.py` ← `ddcolor/pipeline.py`
- `arch_utils/unet.py` ← `basicsr/archs/ddcolor_arch_utils/unet.py`
- `arch_utils/convnext.py` ← `basicsr/archs/ddcolor_arch_utils/convnext.py`
- `arch_utils/transformer_utils.py` ← `basicsr/archs/ddcolor_arch_utils/transformer_utils.py`
- `arch_utils/position_encoding.py` ← `basicsr/archs/ddcolor_arch_utils/position_encoding.py`

## Local modifications

Imports of `basicsr.archs.ddcolor_arch_utils.*` are rewritten to relative
imports (`.arch_utils.*`). Nothing else is changed.

Upstream's modules ship inside a `basicsr` package that is a *different*
project from the PyPI `basicsr==1.4.2` this backend installs for GFPGAN and
facexlib. Installing DDColor with pip would overwrite it, so the six
inference modules are vendored here instead.

Refresh with:

```
python backend/ai/install_ddcolor.py --force
```
