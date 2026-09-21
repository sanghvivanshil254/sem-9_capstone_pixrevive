# PixRevive Image Restoration — Complete Chaptered Implementation Plan

This plan is for transforming your current PixRevive system from:

```text
Upload
   ↓
OpenCV damage detection
   ↓
Telea inpainting
   ↓
Real-ESRGAN
   ↓
CLAHE
   ↓
Sharpening
   ↓
ECCV16 Colorization
```

into a proper **AI image restoration system with intelligent model routing, region-aware restoration, measurable quality testing, and fallback handling**.

---

# Chapter 0 — Current System Audit and Baseline

Before changing anything, freeze the current implementation.

## [x] 0.1 Create a baseline branch

```text
main
│
├── current-baseline
│
└── restoration-v2
```

Do not immediately rewrite the current pipeline.

Your existing system must remain available for comparison.

---

## [x] 0.2 Record the current pipeline

Create:

```text
docs/
└── current_pipeline.md
```

Document:

```text
Input
  ↓
File validation
  ↓
Image decoding
  ↓
Optional denoising
  ↓
OpenCV damage detection
  ↓
Telea inpainting
  ↓
Real-ESRGAN
  ↓
CLAHE
  ↓
Sharpening
  ↓
ECCV16 colorization
  ↓
JPEG output
```

Record:

* model name
* model version
* input resolution
* output resolution
* execution time
* memory usage
* device used
* output file size

---

## [x] 0.3 Create a test dataset

Do not develop restoration algorithms using only one image.

You need a dataset:

```text
tests/
└── restoration_dataset/
    │
    ├── clean/
    │   ├── portrait_001.jpg
    │   ├── landscape_001.jpg
    │   └── family_001.jpg
    │
    ├── damaged/
    │   ├── scratch_001.jpg
    │   ├── stain_001.jpg
    │   └── severe_001.jpg
    │
    └── expected/
        ├── expected_001.jpg
        └── expected_002.jpg
```

You should have at least:

### Category A — Clean images

```text
5–10 images
```

### Category B — Synthetic damage

Take clean images and artificially add:

* scratches
* dust
* random white spots
* black spots
* stains
* color contamination
* missing rectangular regions

This is extremely important because you know the original ground truth.

```text
Clean Image
     ↓
Artificial Damage
     ↓
Restoration
     ↓
Compare with Original
```

This allows objective testing.

---

# [x] Chapter 1 — Refactor the Backend Architecture

Your current `inference.py` is doing too much.

The target structure should be:

```text
backend/
│
├── ai/
│   │
│   ├── analysis/
│   │   ├── quality_analyzer.py
│   │   ├── face_detector.py
│   │   ├── image_type_classifier.py
│   │   └── region_analyzer.py
│   │
│   ├── damage/
│   │   ├── classical_detector.py
│   │   ├── neural_detector.py
│   │   ├── damage_classifier.py
│   │   ├── mask_generator.py
│   │   └── mask_refiner.py
│   │
│   ├── restoration/
│   │   ├── telea_restorer.py
│   │   ├── lama_restorer.py
│   │   ├── diffusion_restorer.py
│   │   ├── face_restorer.py
│   │   └── texture_restorer.py
│   │
│   ├── colorization/
│   │   ├── ddcolor.py
│   │   ├── eccv16.py
│   │   └── colorization_pipeline.py
│   │
│   ├── enhancement/
│   │   ├── realesrgan.py
│   │   ├── sharpening.py
│   │   ├── denoising.py
│   │   └── grain_matching.py
│   │
│   └── pipeline/
│       ├── restoration_pipeline.py
│       ├── pipeline_router.py
│       ├── pipeline_context.py
│       └── pipeline_config.py
│
├── models/
│   ├── damage/
│   ├── lama/
│   ├── face/
│   ├── colorization/
│   └── super_resolution/
│
├── services/
│   ├── image_service.py
│   ├── model_service.py
│   └── quality_service.py
│
└── routers/
    └── image_router.py
```

### Important rule

Each model should have one responsibility.

Bad:

```python
ImageEnhancer()
```

doing everything.

Better:

```python
DamageDetector()
LamaRestorer()
FaceRestorer()
Colorizer()
SuperResolution()
```

Then:

```python
RestorationPipeline()
```

coordinates them.

---

# [x] Chapter 2 — Build a Pipeline Context

Every stage should share a common object.

Conceptually:

```text
PipelineContext
│
├── original_image
├── working_image
├── grayscale_image
├── damage_masks
├── face_regions
├── image_metadata
├── quality_scores
├── processing_log
└── model_outputs
```

Example:

```text
context
│
├── original
│
├── normalized
│
├── analysis
│   ├── is_grayscale
│   ├── blur_score
│   ├── noise_score
│   └── damage_percentage
│
├── masks
│   ├── scratch
│   ├── stain
│   ├── flake
│   └── face_damage
│
└── results
    ├── structural_restore
    ├── colorized
    └── final
```

This makes debugging much easier.

If the final image is wrong, you can inspect:

```text
original
→ mask
→ inpainting
→ face restoration
→ colorization
→ final
```

instead of guessing where the problem happened.

---

# [x] Chapter 3 — Image Input Normalization

Before AI processing:

```text
Input
 ↓
Validate
 ↓
Decode
 ↓
Convert to RGB
 ↓
Correct orientation
 ↓
Remove alpha if required
 ↓
Normalize
```

Create:

```text
ai/preprocessing/
└── image_normalizer.py
```

Handle:

* JPEG
* PNG
* WebP
* grayscale images
* RGB images
* RGBA images
* unusual color profiles
* EXIF orientation

---

## Test Cases

### Test 3.1 — JPEG

```text
Input: normal.jpg
Expected: successful processing
```

### Test 3.2 — PNG with transparency

```text
Input: transparent.png
Expected: alpha handled correctly
```

### Test 3.3 — grayscale image

```text
Input: gray.png
Expected: converted to valid processing format
```

### Test 3.4 — corrupted image

```text
Input: broken.jpg
Expected:
HTTP 400
Clear error message
No server crash
```

### Test 3.5 — oversized image

```text
Input: 20,000 × 20,000 image
Expected:
safe rejection or controlled resizing
no Out-Of-Memory crash
```

---

# [x] Chapter 4 — Image Quality Analysis

Create:

```text
quality_analyzer.py
```

It should calculate:

```text
resolution
aspect ratio
blur score
noise score
brightness
contrast
saturation
grayscale probability
estimated damage percentage
```

The output should look conceptually like:

```json
{
  "width": 2048,
  "height": 2048,
  "blur_score": 0.42,
  "noise_score": 0.73,
  "damage_percentage": 0.18,
  "is_grayscale": false,
  "has_face": true,
  "face_count": 1
}
```

This analysis determines which pipeline to execute.

---

# [x] Chapter 5 — Multi-Stage Damage Detection

This is the most important chapter.

Do not depend only on brightness differences.

Use three layers.

---

## Layer 1 — Classical Computer Vision

Use:

* local contrast
* morphological operations
* edge detection
* color anomaly detection
* thresholding

Output:

```text
classical_damage_mask
```

This is fast.

---

## Layer 2 — Neural Damage Segmentation

Use a segmentation model.

Recommended initial implementation:

```text
U-Net
```

Later:

```text
SegFormer
```

Output:

```text
neural_damage_mask
```

---

## Layer 3 — Mask Fusion

Combine:

```text
classical mask
+
neural mask
```

But do not blindly OR them.

Use confidence:

```text
final_damage_probability =
0.7 × neural_probability
+
0.3 × classical_probability
```

Then:

```text
if probability > threshold:
    damaged
else:
    preserve
```

---

# Chapter 6 — Multi-Class Damage Classification

Do not produce only:

```text
damaged
clean
```

Use:

```text
0 = clean
1 = scratch
2 = dust
3 = white flake
4 = stain
5 = missing region
6 = color contamination
```

Your mask should conceptually be:

```text
damage_map[y][x] = class_id
```

This enables intelligent routing.

Example:

```text
scratch
    ↓
Telea / LaMa

large missing region
    ↓
LaMa

face damage
    ↓
Face pipeline

color contamination
    ↓
LAB color correction
```

---

# [x] Chapter 7 — Mask Refinement

Raw AI masks are usually imperfect.

Create:

```text
mask_refiner.py
```

Operations:

```text
Raw Mask
   ↓
Remove tiny isolated noise
   ↓
Close small gaps
   ↓
Fill holes
   ↓
Connected component filtering
   ↓
Controlled dilation
   ↓
Final mask
```

The mask should have:

```text
binary mask
```

and:

```text
confidence mask
```

For example:

```text
damage mask:
0 = clean
1 = damaged
```

Confidence:

```text
0.0 = uncertain
1.0 = highly confident
```

---

# [x] Chapter 8 — Preserve Known Pixels

This should become a core PixRevive principle.

```text
Original image
       ↓
Damage mask
       ↓
Only damaged regions are allowed to change
```

For every pixel:

```text
if pixel is clean:
    preserve original pixel
else:
    use restored pixel
```

Conceptually:

```text
final =
original × (1 - mask)
+
restored × mask
```

This prevents the AI from unnecessarily changing:

* eyes
* nose
* mouth
* clothing
* background
* facial identity

---

## Test Cases

### Test 8.1 — Clean image

Expected:

```text
Output ≈ Input
```

The restoration model must not alter a clean image dramatically.

---

### Test 8.2 — One scratch

Expected:

```text
Scratch removed
Everything else preserved
```

---

### Test 8.3 — Damage beside eye

Expected:

```text
Damage repaired
Original eye preserved
```

This is a critical test.

---

# [x] Chapter 9 — Small Damage Restoration

For tiny damage:

```text
damage area < threshold
```

Use:

```text
OpenCV Telea
```

or:

```text
OpenCV Navier-Stokes
```

Example:

```text
tiny scratch
     ↓
Telea
```

Why?

Because using a generative model for a tiny scratch is unnecessary and may hallucinate content.

---

# [x] Chapter 10 — General Large-Area Restoration

For large damaged regions:

```text
LaMa
```

Use it for:

* background
* clothing
* wall
* sky
* texture
* large scratches
* irregular missing regions

Pipeline:

```text
Image
  ↓
Damage Mask
  ↓
Mask dilation
  ↓
LaMa
  ↓
Blend with original
  ↓
Preserve clean pixels
```

---

## Test Cases

### Test 10.1 — Large background hole

Expected:

```text
No visible rectangular boundary
```

### Test 10.2 — Repeating fabric

Expected:

```text
Texture continuity
```

### Test 10.3 — Complex background

Expected:

```text
No obvious repeated pattern
```

---

# [x] Chapter 11 — Face-Aware Processing

The face must be isolated.

Pipeline:

```text
Image
 ↓
Face Detection
 ↓
Face Bounding Box
 ↓
Face Landmarks
 ↓
Damage Mask Intersection
```

Then determine:

```text
Is the eye damaged?
Is the nose damaged?
Is the mouth damaged?
Is the forehead damaged?
```

Do not restore the whole face if only the forehead is damaged.

---

## Region priority

```text
Eyes
  ↓
Mouth
  ↓
Nose
  ↓
Face outline
  ↓
Skin
```

The model should protect intact semantic features.

---

# [x] Chapter 12 — Face Restoration

Use:

```text
CodeFormer
```

for severely damaged faces.

Use:

```text
GFPGAN
```

for facial detail enhancement.

The process:

```text
Face Crop
    ↓
Upscale
    ↓
Repair
    ↓
Face Restoration
    ↓
Blend
```

Do not replace the entire original face blindly.

Use:

```text
restored face
+
original face
```

with mask-controlled blending.

---

## Critical face tests

### Test 12.1 — Intact face

Expected:

```text
Identity preserved
```

### Test 12.2 — Damaged forehead

Expected:

```text
Forehead repaired
Eyes unchanged
```

### Test 12.3 — Damaged eye

Expected:

```text
Eye reconstructed naturally
No asymmetry
```

### Test 12.4 — Severe face damage

Expected:

```text
Face remains plausible
No obvious artificial face replacement
```

---

# [x] Chapter 13 — Color Contamination Removal

Your example contains severe red/orange damage.

This must be treated separately.

Pipeline:

```text
RGB
 ↓
LAB
 ↓
Analyze A/B channels
 ↓
Detect abnormal color regions
 ↓
Create contamination mask
 ↓
Correct only contaminated regions
```

Important:

Do not simply desaturate the entire image.

Bad:

```text
image → grayscale
```

because legitimate color is lost.

Better:

```text
abnormal red region → correction
normal red region → preserve
```

---

# [x] Chapter 14 — Restoration Before Colorization

For old black-and-white or severely color-damaged images:

```text
Input
 ↓
Damage color removal
 ↓
Structural restoration
 ↓
Face restoration
 ↓
Colorization
```

Do not colorize heavily damaged images first.

This prevents:

```text
red damage
    ↓
interpreted as real skin/clothing color
```

---

# [x] Chapter 15 — Colorization

Create a model abstraction:

```python
Colorizer
```

Implement:

```text
ECCV16Colorizer
DDColorizer
DeOldifyColorizer
```

Then:

```text
colorizer = model_registry.get("ddcolor")
```

This allows you to compare models without rewriting the entire pipeline.

---

## Colorization tests

### Test 15.1 — Black-and-white portrait

Expected:

```text
natural skin tone
```

### Test 15.2 — Black-and-white landscape

Expected:

```text
reasonable sky and vegetation colors
```

### Test 15.3 — Sepia image

Expected:

```text
sepia tone does not get incorrectly treated as strong real color
```

### Test 15.4 — Already colored image

Expected:

```text
colorization skipped unless explicitly requested
```

---

# [x] Chapter 16 — Super-Resolution

Only after restoration:

```text
Restoration
   ↓
Colorization
   ↓
Face enhancement
   ↓
Real-ESRGAN
```

Real-ESRGAN is for:

```text
resolution enhancement
```

not:

```text
damage reconstruction
```

Use tile processing for large images.

Test:

```text
512 × 512
1024 × 1024
4000 × 3000
```

Expected:

```text
No Out-Of-Memory crash
```

---

# [x] Chapter 17 — Color and Texture Harmonization

After combining separate restoration results, seams may appear.

Apply:

```text
color harmonization
```

and:

```text
texture harmonization
```

Compare:

```text
restored region statistics
```

with:

```text
surrounding original region statistics
```

Match:

* brightness
* contrast
* saturation
* local texture
* grain

Do not apply aggressive global CLAHE.

Your current CLAHE stage should be carefully tested because it can:

* amplify scratches
* exaggerate noise
* create halos
* make skin unnatural

---

# [x] Chapter 18 — Quality Control Layer

Create:

```text
quality_service.py
```

It should check:

```text
output exists
output is readable
dimensions are valid
no NaN values
no completely black image
no completely white image
no extreme color clipping
```

Also calculate:

```text
PSNR
SSIM
LPIPS
```

where a clean reference exists.

For real historical images without a reference, use:

```text
no-reference quality metrics
```

and manual review.

---

# [x] Chapter 19 — Automatic Pipeline Routing

Instead of always running every model:

```text
Pipeline Router
```

should select the pipeline.

Example:

```text
damage_percentage < 1%
        ↓
Traditional restoration

damage_percentage = 1–15%
        ↓
LaMa restoration

damage_percentage > 15%
        ↓
LaMa + advanced restoration

face detected
        ↓
Face pipeline

black-and-white detected
        ↓
Colorization

already colored
        ↓
Skip colorization
```

This saves:

* processing time
* memory
* GPU resources

---

# [x] Chapter 20 — Complete Test Matrix

This is the part many projects completely neglect.

You should test the following.

---

## Group A — File Input Tests

| Test | Input              | Expected            |
| ---- | ------------------ | ------------------- |
| A1   | JPEG               | Success             |
| A2   | PNG                | Success             |
| A3   | WebP               | Success             |
| A4   | Corrupted file     | Clear error         |
| A5   | Unsupported format | Rejection           |
| A6   | 0-byte file        | Rejection           |
| A7   | Oversized file     | Safe rejection      |
| A8   | Huge dimensions    | Controlled handling |

---

# Group B — Image Type Tests

| Test | Image           | Expected                   |
| ---- | --------------- | -------------------------- |
| B1   | Clean color     | Minimal change             |
| B2   | Black-and-white | Restoration + colorization |
| B3   | Sepia           | Correct processing         |
| B4   | Very dark       | Enhancement                |
| B5   | Overexposed     | Controlled recovery        |
| B6   | Low resolution  | Super-resolution           |
| B7   | High resolution | Tile processing            |

---

# Group C — Damage Tests

### C1 — Single scratch

```text
Expected:
scratch removed
surrounding image preserved
```

### C2 — Multiple scratches

```text
Expected:
all major scratches repaired
```

### C3 — White flakes

```text
Expected:
flakes removed
underlying texture reconstructed
```

### C4 — Black spots

```text
Expected:
spots removed
```

### C5 — Large missing region

```text
Expected:
no obvious rectangular patch
```

### C6 — Color stains

```text
Expected:
stain corrected
legitimate colors preserved
```

### C7 — Multiple damage types

```text
Expected:
each damage type routed correctly
```

---

# Group D — Face Tests

### D1 — Clean face

```text
Expected:
identity preserved
```

### D2 — Scratch across forehead

```text
Expected:
forehead repaired
eyes untouched
```

### D3 — Scratch across eye

```text
Expected:
eye reconstruction
no artificial asymmetry
```

### D4 — Missing cheek

```text
Expected:
skin texture reconstructed
```

### D5 — Severe face damage

```text
Expected:
plausible face
identity distortion minimized
```

### D6 — Multiple faces

```text
Expected:
each face independently processed
```

---

# Group E — Color Tests

### E1 — No color damage

```text
Expected:
colors preserved
```

### E2 — Red contamination

```text
Expected:
red artifact reduced
legitimate red preserved
```

### E3 — Sepia

```text
Expected:
correct colorization
```

### E4 — Partial color image

```text
Expected:
existing color retained
missing color reconstructed
```

---

# Group F — Performance Tests

Measure:

```text
processing time
CPU RAM
GPU VRAM
model loading time
output size
```

Test:

```text
512 × 512
1024 × 1024
2048 × 2048
4000 × 3000
```

Expected:

```text
no crash
no memory leak
predictable processing time
```

---

# Group G — API Tests

Test:

```text
POST /api/upload
POST /api/process
```

with:

```text
restore=true
repair=true
colorize=true
denoise=true
```

Then combinations:

```text
restore=false
repair=true
colorize=false
denoise=false
```

Also:

```text
device=cpu
device=gpu
device=auto
```

Expected:

```text
every combination behaves correctly
```

---

# [x] Chapter 21 — Regression Testing

Every time you change a model, run:

```text
test suite
```

The output should compare:

```text
old result
vs
new result
```

You should store:

```text
test_results/
│
├── baseline/
│
├── v2/
│
└── comparison/
```

Compare:

```text
PSNR
SSIM
LPIPS
processing time
memory usage
```

Do not judge improvements only by:

> "This image looks better to me."

That is subjective.

---

# [x] Chapter 22 — Visual Debugging Dashboard

Add a development-only mode that outputs:

```text
1. Original
2. Classical mask
3. Neural mask
4. Final mask
5. LaMa output
6. Face crop
7. Face restoration
8. Colorized output
9. Final result
```

Example:

```text
debug_output/
├── 01_original.png
├── 02_classical_mask.png
├── 03_neural_mask.png
├── 04_final_mask.png
├── 05_structural_restore.png
├── 06_face_crop.png
├── 07_face_restore.png
├── 08_colorized.png
└── 09_final.png
```

This will dramatically speed up debugging.

---

# [x] Chapter 23 — Recommended Development Order

Do not implement everything at once.

## Phase 1 — Refactoring

```text
Week 1
```

Implement:

```text
PipelineContext
Model interfaces
Model registry
Logging
```

---

## Phase 2 — Damage Detection

```text
Week 2
```

Implement:

```text
classical detector
mask generator
mask refinement
visual mask debugging
```

Do not add complex AI restoration yet.

---

## Phase 3 — LaMa

```text
Week 3
```

Implement:

```text
LaMa integration
mask-based restoration
original pixel preservation
```

Test extensively.

---

## Phase 4 — Face Pipeline

```text
Week 4
```

Implement:

```text
face detection
face cropping
CodeFormer/GFPGAN
face blending
```

---

## Phase 5 — Color Pipeline

```text
Week 5
```

Implement:

```text
color contamination detection
LAB correction
DDColor or selected colorizer
```

---

## Phase 6 — Super-Resolution

```text
Week 6
```

Move:

```text
Real-ESRGAN
```

to the correct position near the end.

---

## Phase 7 — Quality and Testing

```text
Week 7
```

Implement:

```text
automated tests
quality metrics
performance benchmarks
regression testing
```

---

# Final Target Architecture

Your final system should look like this:

```text
                 ┌──────────────┐
                 │ Input Image  │
                 └──────┬───────┘
                        │
                        ▼
                ┌───────────────┐
                │ Normalization │
                └──────┬────────┘
                       │
                       ▼
                ┌───────────────┐
                │ Image Analysis│
                └──────┬────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Damage        Face        Image Type
       Analysis      Analysis     Analysis
          │            │            │
          └────────────┼────────────┘
                       ▼
                ┌───────────────┐
                │ Mask Refining │
                └──────┬────────┘
                       │
                       ▼
                ┌───────────────┐
                │ Region Router │
                └──────┬────────┘
                       │
          ┌────────────┼─────────────┐
          ▼            ▼             ▼
       Background     Face       Color Damage
       → LaMa       → Face AI    → LAB Repair
          │            │             │
          └────────────┼─────────────┘
                       ▼
                Structural Image
                       │
                       ▼
                  Colorization
                       │
                       ▼
                Face Enhancement
                       │
                       ▼
                  Real-ESRGAN
                       │
                       ▼
              Color Harmonization
                       │
                       ▼
                 Grain Matching
                       │
                       ▼
                Quality Validation
                       │
                       ▼
                    OUTPUT
```

## The most important implementation rule

**Do not start by downloading five new models.**

Start by implementing:

```text
1. PipelineContext
2. Damage mask system
3. Mask visualization
4. Region-based routing
5. Original pixel preservation
6. Automated test dataset
```

Then integrate:

```text
LaMa
→ Face Restoration
→ Colorization
→ Real-ESRGAN
```

in that order.

The current biggest weakness is **not lack of models**. It is lack of **controlled routing, damage classification, preservation logic, and objective testing**. If you fix those four things, the existing PixRevive architecture becomes a solid foundation instead of a chain of independent enhancement filters.
