import os
import time
import json
import numpy as np
from PIL import Image
import sys

# Ensure backend directory is in python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ai.pipeline.restoration_pipeline import RestorationPipeline
from tests.metrics import calculate_psnr, calculate_ssim

def run_tests():
    print("Initializing Restoration Pipeline...")
    pipeline = RestorationPipeline()
    
    test_dir = os.path.join(os.path.dirname(__file__), "test_data")
    input_dir = os.path.join(test_dir, "input")
    ref_dir = os.path.join(test_dir, "reference")
    out_dir = os.path.join(test_dir, "output")
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    
    results = {}
    
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        print(f"No test images found in {input_dir}. Please add some images and run again.")
        return
        
    for filename in files:
        print(f"\nProcessing {filename}...")
        input_path = os.path.join(input_dir, filename)
        ref_path = os.path.join(ref_dir, filename)
        out_path = os.path.join(out_dir, filename)
        
        with open(input_path, "rb") as f:
            image_bytes = f.read()
            
        start_time = time.time()
        
        # Run full pipeline with all features on
        output_bytes = pipeline.process(
            image_bytes=image_bytes,
            enable_restoration=True,
            enable_repair=True,
            enable_colorization=True,
            enable_denoise=True,
            device_preference="auto"
        )
        
        elapsed = time.time() - start_time
        
        # Save output
        with open(out_path, "wb") as f:
            f.write(output_bytes)
            
        print(f"Saved to {out_path} in {elapsed:.2f} seconds.")
        
        img_stats = {
            "processing_time_sec": round(elapsed, 2)
        }
        
        # Calculate metrics if reference exists
        if os.path.exists(ref_path):
            out_img = np.array(Image.open(out_path).convert("RGB"))
            ref_img = np.array(Image.open(ref_path).convert("RGB"))
            
            # Ensure dimensions match for PSNR/SSIM
            if out_img.shape == ref_img.shape:
                psnr = calculate_psnr(ref_img, out_img)
                ssim = calculate_ssim(ref_img, out_img)
                if psnr is not None:
                    img_stats["psnr"] = round(psnr, 2)
                if ssim is not None:
                    img_stats["ssim"] = round(ssim, 4)
                print(f"Metrics -> PSNR: {psnr}, SSIM: {ssim}")
            else:
                print(f"Reference shape {ref_img.shape} does not match output shape {out_img.shape}. Cannot compute metrics.")
                
        results[filename] = img_stats
        
    results_path = os.path.join(out_dir, "test_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nAll tests complete. Results saved to {results_path}")

if __name__ == "__main__":
    run_tests()
