from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends, Header
from pydantic import BaseModel
import logging
import os
import shutil
import time
import uuid
import json
import zipfile
import tempfile
import re
from typing import Optional, List

logger = logging.getLogger(__name__)


def verify_admin_token(x_admin_token: Optional[str] = Header(None)) -> None:
    """Validate admin token header if PIXREVIVE_ADMIN_TOKEN environment variable is set."""
    required_token = os.getenv("PIXREVIVE_ADMIN_TOKEN", "").strip()
    if required_token and (not x_admin_token or x_admin_token != required_token):
        raise HTTPException(status_code=401, detail="Invalid or missing admin token.")


router = APIRouter(dependencies=[Depends(verify_admin_token)])

# Canonical weight directory (shared by all models)
AI_WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ai", "weights")

# Training data paths
TRAINING_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "training_data")
REGISTRY_PATH = os.path.join(TRAINING_DIR, "registry.json")
BW_DIR = os.path.join(TRAINING_DIR, "bw")
COLOR_DIR = os.path.join(TRAINING_DIR, "color")

# Upload ceilings. Without these, shutil.copyfileobj streams an unbounded body
# straight to disk and a single request can fill the volume.
MAX_WEIGHTS_UPLOAD_BYTES = 1024 * 1024 * 1024   # 1 GB — GFPGAN/RealESRGAN are ~65-350 MB
MAX_ZIP_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024   # 2 GB compressed
MAX_ZIP_EXTRACTED_BYTES = 8 * 1024 * 1024 * 1024  # 8 GB uncompressed (zip-bomb guard)

# Global store to track background ZIP uploads progress
# Keys: upload_id, Values: {status, progress, message, added_count}
upload_progress_store = {}
# Entries are only ever added, so prune finished ones — a long-lived server would
# otherwise accumulate one dict per upload forever.
_UPLOAD_PROGRESS_TTL_SECONDS = 3600


def _set_progress(upload_id: str, **fields) -> None:
    """Record progress for `upload_id` and prune stale entries."""
    fields["updated_at"] = time.time()
    upload_progress_store[upload_id] = fields

    now = time.time()
    stale = [
        uid for uid, entry in upload_progress_store.items()
        if uid != upload_id
        and entry.get("status") in ("completed", "failed")
        and now - entry.get("updated_at", now) > _UPLOAD_PROGRESS_TTL_SECONDS
    ]
    for uid in stale:
        upload_progress_store.pop(uid, None)


def _save_upload_bounded(upload_file: UploadFile, dest_path: str, max_bytes: int) -> int:
    """Stream `upload_file` to `dest_path`, aborting past `max_bytes`.

    Returns the number of bytes written. Raises HTTPException(413) and removes
    the partial file if the limit is exceeded.
    """
    written = 0
    chunk_size = 1024 * 1024
    try:
        with open(dest_path, "wb") as buffer:
            while True:
                chunk = upload_file.file.read(chunk_size)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum is {max_bytes // (1024 * 1024)} MB.",
                    )
                buffer.write(chunk)
    except Exception:
        try:
            os.remove(dest_path)
        except OSError:
            pass
        raise
    return written


def _quiet_remove(path: str) -> None:
    """Delete `path` if present, ignoring failures."""
    try:
        os.remove(path)
    except OSError:
        pass


def _looks_like_torch_checkpoint(path: str) -> bool:
    """Cheap structural check that `path` is a torch.save() artifact.

    torch.save writes a ZIP container ("PK\\x03\\x04") in the modern format, or a
    legacy pickle beginning with the protocol-2 opcode (0x80 0x02). This rejects
    an image or text file renamed to .pth; it does NOT make an untrusted
    checkpoint safe — torch.load still unpickles, which can execute code.
    """
    try:
        with open(path, "rb") as fp:
            head = fp.read(4)
    except OSError:
        return False
    return head[:4] == b"PK\x03\x04" or head[:2] == b"\x80\x02"


class BatchDeleteRequest(BaseModel):
    ids: List[str]

def load_registry():
    if not os.path.exists(REGISTRY_PATH):
        return {"pairs": []}
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"pairs": []}

def save_registry(data):
    os.makedirs(TRAINING_DIR, exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def async_process_zip_dataset(zip_file_path: str, upload_id: str):
    temp_dir = tempfile.mkdtemp()
    try:
        _set_progress(
            upload_id,
            status="processing",
            progress=15,
            message="Extracting ZIP archive...",
        )
        
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            infolist = zip_ref.infolist()
            total_members = len(infolist)
            safe_root = os.path.abspath(temp_dir) + os.sep
            extracted_bytes = 0

            abs_temp_dir = os.path.abspath(temp_dir)
            for idx, member in enumerate(infolist):
                target_path = os.path.abspath(os.path.join(temp_dir, member.filename))
                try:
                    is_safe = os.path.commonpath([abs_temp_dir, target_path]) == abs_temp_dir
                except (ValueError, OSError):
                    is_safe = False

                if not is_safe or not target_path.startswith(safe_root):
                    logger.warning("Skipping unsafe zip member: %r", member.filename)
                    continue

                # Zip-bomb guard: a few-MB archive can expand to hundreds of GB.
                extracted_bytes += member.file_size
                if extracted_bytes > MAX_ZIP_EXTRACTED_BYTES:
                    raise ValueError(
                        "Archive expands beyond the "
                        f"{MAX_ZIP_EXTRACTED_BYTES // (1024 ** 3)} GB uncompressed limit."
                    )

                zip_ref.extract(member, temp_dir)

                # Periodically update extraction progress (range 15% -> 40%)
                if idx % max(1, total_members // 10) == 0:
                    pct = 15 + int((idx / max(total_members, 1)) * 25)
                    _set_progress(
                        upload_id,
                        status="processing",
                        progress=pct,
                        message=f"Extracting files ({idx}/{total_members})...",
                    )
        
        _set_progress(
            upload_id,
            status="processing",
            progress=40,
            message="Scanning directory for image assets...",
        )
        
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        all_images = []
        for root, _, files in os.walk(temp_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in image_extensions:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, temp_dir)
                    all_images.append((full_path, rel_path, file))
        
        _set_progress(
            upload_id,
            status="processing",
            progress=50,
            message=f"Analyzing {len(all_images)} images and pairing...",
        )
        
        bw_candidates = []
        color_candidates = []
        unclassified = []
        
        bw_keywords = ["bw", "gray", "grey", "degraded", "input", "lr"]
        color_keywords = ["color", "clean", "target", "gt", "original", "hr"]
        
        for full_path, rel_path, filename in all_images:
            rel_path_lower = rel_path.lower()
            
            is_bw = any(kw in rel_path_lower for kw in bw_keywords)
            is_color = any(kw in rel_path_lower for kw in color_keywords)
            
            if is_bw and not is_color:
                bw_candidates.append((full_path, rel_path, filename))
            elif is_color and not is_bw:
                color_candidates.append((full_path, rel_path, filename))
            else:
                unclassified.append((full_path, rel_path, filename))
                
        def clean_path(rel_path):
            parts = rel_path.replace("\\", "/").split("/")
            cleaned_parts = [p for p in parts if not any(kw in p.lower() for kw in bw_keywords + color_keywords)]
            return "/".join(cleaned_parts)

        pairs_to_add = []
        
        bw_map = {}
        for item in bw_candidates:
            cpath = clean_path(item[1])
            bw_map[cpath] = item
            
        color_map = {}
        for item in color_candidates:
            cpath = clean_path(item[1])
            color_map[cpath] = item
            
        matched_cpaths = set(bw_map.keys()) & set(color_map.keys())
        for cpath in matched_cpaths:
            pairs_to_add.append((bw_map[cpath][0], color_map[cpath][0], os.path.basename(cpath)))
            bw_map.pop(cpath)
            color_map.pop(cpath)
            
        remaining_items = list(bw_map.values()) + list(color_map.values()) + unclassified
        prefix_map = {}
        suffix_pattern = re.compile(r"[_-](bw|gray|grey|degraded|input|lr|color|clean|target|gt|original|hr)$", re.IGNORECASE)
        
        for full_path, rel_path, filename in remaining_items:
            base, ext = os.path.splitext(filename)
            base_clean = suffix_pattern.sub("", base)
            if base_clean not in prefix_map:
                prefix_map[base_clean] = []
            prefix_map[base_clean].append((full_path, filename))
            
        for base_clean, items in prefix_map.items():
            if len(items) == 2:
                item1, item2 = items[0], items[1]
                is_i1_bw = any(kw in item1[1].lower() for kw in bw_keywords)
                is_i2_color = any(kw in item2[1].lower() for kw in color_keywords)
                
                if is_i1_bw or is_i2_color:
                    pairs_to_add.append((item1[0], item2[0], base_clean + ".png"))
                else:
                    pairs_to_add.append((item2[0], item1[0], base_clean + ".png"))
                    
        registry = load_registry()
        total_pairs = len(pairs_to_add)
        
        _set_progress(
            upload_id,
            status="processing",
            progress=70,
            message=f"Found {total_pairs} paired images. Saving and copying...",
        )
        
        os.makedirs(BW_DIR, exist_ok=True)
        os.makedirs(COLOR_DIR, exist_ok=True)
        
        added_count = 0
        for idx, (bw_path, color_path, base_name) in enumerate(pairs_to_add):
            pair_id = f"pair_{uuid.uuid4().hex[:12]}"
            bw_ext = os.path.splitext(bw_path)[1].lower() or ".png"
            color_ext = os.path.splitext(color_path)[1].lower() or ".png"
            
            bw_filename = f"{pair_id}_bw{bw_ext}"
            color_filename = f"{pair_id}_color{color_ext}"
            
            dest_bw = os.path.join(BW_DIR, bw_filename)
            dest_color = os.path.join(COLOR_DIR, color_filename)
            
            shutil.copy2(bw_path, dest_bw)
            shutil.copy2(color_path, dest_color)
            
            registry["pairs"].append({
                "id": pair_id,
                "bw_name": bw_filename,
                "color_name": color_filename,
                "original_name": base_name
            })
            added_count += 1
            
            # Update save progress periodically (70% -> 95%)
            if idx % max(1, total_pairs // 10) == 0:
                pct = 70 + int((idx / total_pairs) * 25)
                _set_progress(
                    upload_id,
                    status="processing",
                    progress=pct,
                    message=f"Registering and writing image file pairs ({idx}/{total_pairs})...",
                )
            
        save_registry(registry)
        
        _set_progress(
            upload_id,
            status="completed",
            progress=100,
            message=f"Successfully processed dataset and added {added_count} training image pairs.",
            added_count=added_count,
        )
        
    except Exception as e:
        _set_progress(
            upload_id,
            status="failed",
            progress=0,
            message=f"Failed to process ZIP file dataset: {str(e)}",
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        if os.path.exists(zip_file_path):
            try:
                os.remove(zip_file_path)
            except Exception:
                pass

# Admin weight upload targets (plan 6.2): every model the pipeline can load
# at runtime, mapped to its canonical file layout under backend/ai/weights/.
# Subdirectory targets (DDColor, Restormer deblur) are created on demand.
_WEIGHT_UPLOAD_TARGETS = {
    "realesrgan": ("RealESRGAN_x4plus.pth",   AI_WEIGHTS_DIR),
    "gfpgan":     ("GFPGANv1.4.pth",          AI_WEIGHTS_DIR),
    "lama":       ("big-lama.pt",             AI_WEIGHTS_DIR),
    "colorizer":  ("ddcolor_modelscope.pth",  os.path.join(AI_WEIGHTS_DIR, "ddcolor")),
    "deblur":     ("motion_deblurring.pth",   os.path.join(AI_WEIGHTS_DIR, "restormer")),
}

@router.post("/weights/upload")
async def upload_weights(
    file: UploadFile = File(...),
    model_type: str = Form(...)
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")
    # LaMa ships as .pt; every other model as .pth — both are torch checkpoints.
    if not (file.filename.endswith(".pth") or file.filename.endswith(".pt")):
        raise HTTPException(status_code=400, detail="Only PyTorch .pth/.pt files are allowed.")

    target = _WEIGHT_UPLOAD_TARGETS.get(model_type)
    if target is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid model type. Use one of: " + ", ".join(sorted(_WEIGHT_UPLOAD_TARGETS)) + ".",
        )
    target_filename, target_dir = target

    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, target_filename)
    # Stage to a sibling temp file, then atomically replace. Writing straight to
    # target_path destroys the working weights the moment the upload is aborted
    # or exceeds the size limit, leaving the model permanently unloadable.
    staging_path = f"{target_path}.upload-{uuid.uuid4().hex[:8]}"

    try:
        size = _save_upload_bounded(file, staging_path, MAX_WEIGHTS_UPLOAD_BYTES)
        if size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if not _looks_like_torch_checkpoint(staging_path):
            raise HTTPException(
                status_code=400,
                detail="File is not a recognisable PyTorch checkpoint.",
            )
        os.replace(staging_path, target_path)
    except HTTPException:
        _quiet_remove(staging_path)
        raise
    except Exception as e:
        _quiet_remove(staging_path)
        raise HTTPException(status_code=500, detail=f"Failed to save weight file: {str(e)}")

    # Hot-reload is not supported — the running process keeps the old weights.
    logger.info("Weight file saved to %s (%d bytes). Restart backend to apply.", target_path, size)
    return {
        "message": f"Saved {target_filename}. Restart the backend to apply the new weights.",
        "status": "success",
        "size_bytes": size,
        "requires_restart": True,
    }

@router.post("/training-data/upload")
async def upload_training_data_zip(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    if not file.filename or not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only ZIP files are allowed.")

    upload_id = f"upload_{uuid.uuid4().hex[:8]}"
    _set_progress(
        upload_id,
        status="processing",
        progress=5,
        message="Saving ZIP archive upload to disk...",
    )

    os.makedirs(TRAINING_DIR, exist_ok=True)
    temp_zip_path = os.path.join(TRAINING_DIR, f"temp_{upload_id}.zip")

    try:
        _save_upload_bounded(file, temp_zip_path, MAX_ZIP_UPLOAD_BYTES)
        if not zipfile.is_zipfile(temp_zip_path):
            raise HTTPException(status_code=400, detail="File is not a valid ZIP archive.")
    except HTTPException as exc:
        _quiet_remove(temp_zip_path)
        _set_progress(upload_id, status="failed", progress=0, message=str(exc.detail))
        raise
    except Exception as e:
        _quiet_remove(temp_zip_path)
        _set_progress(upload_id, status="failed", progress=0, message=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to save ZIP upload: {str(e)}")

    _set_progress(
        upload_id,
        status="processing",
        progress=10,
        message="Upload complete. Starting background extraction task...",
    )
    
    # Enqueue background processing
    background_tasks.add_task(async_process_zip_dataset, temp_zip_path, upload_id)
    
    return {
        "upload_id": upload_id,
        "status": "processing",
        "message": "Upload complete. Extracting dataset in background."
    }

@router.get("/training-data/upload-progress")
async def get_upload_progress(upload_id: str):
    if upload_id not in upload_progress_store:
        raise HTTPException(status_code=404, detail="Upload process ID not found.")
    return upload_progress_store[upload_id]

@router.get("/training-data/pairs")
async def get_training_pairs(page: int = 1, limit: int = 10, q: Optional[str] = None):
    registry = load_registry()
    pairs = registry.get("pairs", [])
    
    # Reverse to show newest uploads first
    pairs = list(reversed(pairs))
    
    # Apply search filter if present
    if q and q.strip():
        search_query = q.lower().strip()
        pairs = [
            p for p in pairs 
            if search_query in p.get("original_name", "").lower() or search_query in p.get("id", "").lower()
        ]
    
    total = len(pairs)
    total_pages = (total + limit - 1) // limit if total > 0 else 1
    
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    
    paginated_pairs = pairs[start_idx:end_idx]
    
    for p in paginated_pairs:
        p["bw_url"] = f"/static/training_data/bw/{p['bw_name']}"
        p["color_url"] = f"/static/training_data/color/{p['color_name']}"
        
    return {
        "pairs": paginated_pairs,
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": total_pages
    }

@router.post("/training-data/pairs")
async def create_training_pair(
    bw_file: UploadFile = File(...),
    color_file: UploadFile = File(...)
):
    allowed_exts = {".jpg", ".jpeg", ".png", ".webp"}
    bw_name = bw_file.filename or ""
    color_name = color_file.filename or ""
    bw_ext = os.path.splitext(bw_name)[1].lower()
    color_ext = os.path.splitext(color_name)[1].lower()
    
    if bw_ext not in allowed_exts or color_ext not in allowed_exts:
        raise HTTPException(status_code=400, detail="Only JPG, JPEG, PNG, and WebP images are allowed.")
        
    pair_id = f"pair_{uuid.uuid4().hex[:12]}"
    bw_filename = f"{pair_id}_bw{bw_ext}"
    color_filename = f"{pair_id}_color{color_ext}"
    
    os.makedirs(BW_DIR, exist_ok=True)
    os.makedirs(COLOR_DIR, exist_ok=True)
    
    bw_path = os.path.join(BW_DIR, bw_filename)
    color_path = os.path.join(COLOR_DIR, color_filename)
    
    try:
        with open(bw_path, "wb") as f:
            shutil.copyfileobj(bw_file.file, f)
        with open(color_path, "wb") as f:
            shutil.copyfileobj(color_file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save image files: {str(e)}")
        
    registry = load_registry()
    new_pair = {
        "id": pair_id,
        "bw_name": bw_filename,
        "color_name": color_filename,
        "original_name": bw_file.filename
    }
    registry["pairs"].append(new_pair)
    save_registry(registry)
    
    new_pair["bw_url"] = f"/static/training_data/bw/{bw_filename}"
    new_pair["color_url"] = f"/static/training_data/color/{color_filename}"
    
    return new_pair

@router.put("/training-data/pairs/{pair_id}")
async def update_training_pair(
    pair_id: str,
    bw_file: Optional[UploadFile] = File(None),
    color_file: Optional[UploadFile] = File(None)
):
    registry = load_registry()
    pairs = registry.get("pairs", [])
    
    pair_index = -1
    for i, p in enumerate(pairs):
        if p["id"] == pair_id:
            pair_index = i
            break
            
    if pair_index == -1:
        raise HTTPException(status_code=404, detail="Pair not found.")
        
    pair = pairs[pair_index]
    allowed_exts = {".jpg", ".jpeg", ".png", ".webp"}
    
    os.makedirs(BW_DIR, exist_ok=True)
    os.makedirs(COLOR_DIR, exist_ok=True)
    
    if bw_file is not None:
        bw_ext = os.path.splitext(bw_file.filename or "")[1].lower()
        if bw_ext not in allowed_exts:
            raise HTTPException(status_code=400, detail="Only JPG, JPEG, PNG, and WebP images are allowed.")
            
        old_bw_path = os.path.join(BW_DIR, pair["bw_name"])
        if os.path.exists(old_bw_path):
            try:
                os.remove(old_bw_path)
            except Exception:
                pass
            
        new_bw_filename = f"{pair_id}_bw{bw_ext}"
        new_bw_path = os.path.join(BW_DIR, new_bw_filename)
        with open(new_bw_path, "wb") as f:
            shutil.copyfileobj(bw_file.file, f)
        pair["bw_name"] = new_bw_filename
        
    if color_file is not None:
        color_ext = os.path.splitext(color_file.filename or "")[1].lower()
        if color_ext not in allowed_exts:
            raise HTTPException(status_code=400, detail="Only JPG, JPEG, PNG, and WebP images are allowed.")
            
        old_color_path = os.path.join(COLOR_DIR, pair["color_name"])
        if os.path.exists(old_color_path):
            try:
                os.remove(old_color_path)
            except Exception:
                pass
            
        new_color_filename = f"{pair_id}_color{color_ext}"
        new_color_path = os.path.join(COLOR_DIR, new_color_filename)
        with open(new_color_path, "wb") as f:
            shutil.copyfileobj(color_file.file, f)
        pair["color_name"] = new_color_filename
        
    pairs[pair_index] = pair
    save_registry(registry)
    
    pair["bw_url"] = f"/static/training_data/bw/{pair['bw_name']}"
    pair["color_url"] = f"/static/training_data/color/{pair['color_name']}"
    
    return pair

@router.delete("/training-data/pairs/all")
async def delete_all_training_pairs():
    # Remove files from directories
    if os.path.exists(BW_DIR):
        try:
            shutil.rmtree(BW_DIR)
        except Exception:
            pass
    if os.path.exists(COLOR_DIR):
        try:
            shutil.rmtree(COLOR_DIR)
        except Exception:
            pass
            
    # Re-create empty directories
    os.makedirs(BW_DIR, exist_ok=True)
    os.makedirs(COLOR_DIR, exist_ok=True)
    
    # Save empty registry
    save_registry({"pairs": []})
    
    return {"message": "Successfully cleared all training dataset pairs."}

@router.post("/training-data/pairs/batch-delete")
async def batch_delete_pairs(request: BatchDeleteRequest):
    ids_to_delete = set(request.ids)
    if not ids_to_delete:
        return {"message": "No IDs provided", "deleted_count": 0}
        
    registry = load_registry()
    pairs = registry.get("pairs", [])
    
    new_pairs = []
    deleted_count = 0
    
    for p in pairs:
        if p["id"] in ids_to_delete:
            bw_path = os.path.join(BW_DIR, p["bw_name"])
            color_path = os.path.join(COLOR_DIR, p["color_name"])
            if os.path.exists(bw_path):
                try:
                    os.remove(bw_path)
                except Exception:
                    pass
            if os.path.exists(color_path):
                try:
                    os.remove(color_path)
                except Exception:
                    pass
            deleted_count += 1
        else:
            new_pairs.append(p)
            
    registry["pairs"] = new_pairs
    save_registry(registry)
    
    return {"message": f"Successfully deleted {deleted_count} pairs", "deleted_count": deleted_count}

@router.delete("/training-data/pairs/{pair_id}")
async def delete_training_pair(pair_id: str):
    registry = load_registry()
    pairs = registry.get("pairs", [])
    
    pair_index = -1
    for i, p in enumerate(pairs):
        if p["id"] == pair_id:
            pair_index = i
            break
            
    if pair_index == -1:
        raise HTTPException(status_code=404, detail="Pair not found.")
        
    pair = pairs.pop(pair_index)
    
    bw_path = os.path.join(BW_DIR, pair["bw_name"])
    color_path = os.path.join(COLOR_DIR, pair["color_name"])
    
    if os.path.exists(bw_path):
        try:
            os.remove(bw_path)
        except Exception:
            pass
            
    if os.path.exists(color_path):
        try:
            os.remove(color_path)
        except Exception:
            pass
            
    return {"message": "Successfully deleted image pair", "id": pair_id}

# ==========================================
# MODEL STATUS & AUTOMATIC TRAINING PIPELINE
# ==========================================

import time

# Global training pipeline store
training_progress_store = {
    "status": "idle",
    "progress": 0,
    "message": "No active training pipeline.",
    "epochs": [],
    "results": None
}

def async_run_training_pipeline():
    """Placeholder: real training/benchmark is not available in this build (D15).

    The previous unreachable RealESRGAN benchmark body was removed; the
    /training/* endpoints remain as honest no-ops for the UI.
    """
    global training_progress_store
    training_progress_store = {
        "status": "failed",
        "progress": 0,
        "message": "Benchmark training pipeline is not available in this build.",
        "epochs": [],
        "results": None
    }

@router.get("/weights/status")
async def get_weights_status():
    """Return the status of all model weight files in the canonical weights directory."""
    weights_info = {}
    expected_weights = {
        "gfpgan":      "GFPGANv1.4.pth",
        "realesrgan":  "RealESRGAN_x4plus.pth",
        "lama":        "big-lama.pt",
        "deblur":      os.path.join("restormer", "motion_deblurring.pth"),
    }
    for key, filename in expected_weights.items():
        path = os.path.join(AI_WEIGHTS_DIR, filename)
        exists = os.path.exists(path)
        size_mb = round(os.path.getsize(path) / 1_048_576, 1) if exists else 0
        weights_info[key] = {
            "file_name":   filename,
            "file_exists": exists,
            "size_mb":     size_mb,
            "path":        path,
        }

    ddcolor_dir      = os.path.join(AI_WEIGHTS_DIR, "ddcolor")
    ddcolor_weight   = "ddcolor_modelscope.pth"
    ddcolor_path     = os.path.join(ddcolor_dir, ddcolor_weight)
    ddcolor_exists   = os.path.isfile(ddcolor_path) and os.path.getsize(ddcolor_path) > 10_000_000
    ddcolor_size_mb  = round(os.path.getsize(ddcolor_path) / 1_048_576, 1) if ddcolor_exists else 0.0

    weights_info["colorizer"] = {
        "file_name":   ddcolor_weight,
        "file_exists": ddcolor_exists,
        "size_mb":     ddcolor_size_mb,
        "path":        ddcolor_dir,
        "model":       "DDColor (ICCV 2023)",
    }

    # Pipeline readiness
    colorizer_ready  = False
    colorizer_loaded = False
    colorizer_device = "unknown"
    try:
        from services.image_service import pipeline
        colorizer_ready  = pipeline.is_colorizer_ready()
        colorizer_loaded = bool(getattr(pipeline.colorizer, "is_loaded", False))
        colorizer_device = getattr(pipeline.colorizer, "device", "unknown")
    except Exception:
        pass

    weights_info["colorizer"]["loaded"] = colorizer_loaded
    weights_info["colorizer"]["device"] = colorizer_device

    return {
        **weights_info,
        "weights":         weights_info,
        "colorizer_ready": colorizer_ready,
        "ddcolor_ready":   colorizer_ready,   # canonical name
    }

@router.post("/training/start")
async def start_training_pipeline(background_tasks: BackgroundTasks):
    global training_progress_store
    
    if training_progress_store["status"] == "running":
        raise HTTPException(status_code=400, detail="Training pipeline is already running.")
        
    training_progress_store = {
        "status": "running",
        "progress": 5,
        "message": "Initializing training pipeline task...",
        "epochs": [],
        "results": None
    }
    
    background_tasks.add_task(async_run_training_pipeline)
    return {"message": "Training pipeline started.", "status": "started"}

@router.get("/training/status")
async def get_training_status():
    global training_progress_store
    return training_progress_store



