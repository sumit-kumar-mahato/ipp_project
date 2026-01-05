# Quick Start Guide

## 5-Minute Setup

### Step 1: Clone/Extract Project
```bash
cd document-enhancement-compression
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**If you have issues:**
```bash
# For Windows users
python -m pip install --upgrade pip
pip install streamlit opencv-python Pillow matplotlib numpy scikit-image

# For macOS/Linux users
pip3 install streamlit opencv-python Pillow matplotlib numpy scikit-image
```

### Step 3: Run the App
```bash
streamlit run document_app.py
```

The app will open automatically at: `http://localhost:8501`

---

## First Run - Example Workflow

### 1. Prepare a Test Image
- Use any scanned document image
- JPG, PNG, TIFF, BMP supported
- Size: 100KB - 10MB recommended

### 2. Upload Image
- Click "Upload Document Image"
- Select your file
- Wait for processing to start

### 3. Configure Settings (Sidebar)

**For Scanned Documents:**
```
Noise Removal: Median Filter (Kernel: 5)
Enhancement: Histogram Equalization
Binarization: Otsu's Thresholding
Compression: Both (RLE + Huffman)
```

**For Old Documents:**
```
Noise Removal: Gaussian Filter (Kernel: 5, Sigma: 1.0)
Enhancement: CLAHE (Clip: 2.0, Tile: 8)
Binarization: Adaptive Thresholding (Block: 11)
Compression: Huffman
```

### 4. Explore the Tabs

**Preview Tab**
- See original image
- Check dimensions and size

**Processing Tab**
- Watch noise removal effect
- See histogram transformation
- Compare before/after images

**Binarization Tab**
- View binary conversion
- Check foreground/background ratio
- See threshold value used

**Compression Tab**
- Compare compression methods
- View size reduction percentage
- See visual compression comparison

**Analysis & Report Tab**
- Check quality metrics (PSNR, SSIM, MSE)
- Read recommendations
- Download processed images

### 5. Download Results
- Enhanced image (PNG)
- Binary image (PNG)
- Detailed report (TXT)

---

## Common Use Cases

### Use Case 1: Clean Old Scanned Document

**Problem**: Faded, with visible grain noise

**Configuration**:
```
Noise: Gaussian Filter (Kernel: 7, Sigma: 1.5)
Enhancement: CLAHE (Clip: 3.0)
Binarization: Adaptive
Compression: Both
```

**Expected Results**:
- MSE: 100-200
- PSNR: 25-30 dB
- SSIM: 0.85-0.95
- Compression: 70-85% reduction

---

### Use Case 2: Clean Receipt Image

**Problem**: Salt-and-pepper noise, low contrast

**Configuration**:
```
Noise: Median Filter (Kernel: 5)
Enhancement: Histogram Equalization
Binarization: Otsu's
Compression: RLE
```

**Expected Results**:
- MSE: 50-100
- PSNR: 30-35 dB
- SSIM: 0.90-0.98
- Compression: 80-90% reduction (excellent for binary)

---

### Use Case 3: Medical Document

**Problem**: Complex texture, need edge preservation

**Configuration**:
```
Noise: Bilateral Filter (D: 9, Color: 75, Space: 75)
Enhancement: CLAHE (Clip: 2.5)
Binarization: Adaptive (Block: 13)
Compression: Huffman
```

**Expected Results**:
- MSE: 50-150
- PSNR: 28-33 dB
- SSIM: 0.88-0.96
- Compression: 50-70% reduction

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'streamlit'"

**Solution**:
```bash
pip install streamlit --upgrade
```

### Issue: App crashes when uploading large image

**Solution**:
1. Resize image to under 4000×4000 pixels
2. Or use image editor to reduce size first
3. Check available RAM

### Issue: Binarization looks too dark or light

**Solution**:
```
Try switching between:
- Otsu's Thresholding (automatic)
- Adaptive Thresholding (manual block size adjustment)
```

### Issue: Compression ratio too low

**Solution**:
```
Ensure binary image is truly black & white:
- Use Otsu's or Adaptive thresholding
- Check "Binarization" tab for proper conversion
- Some documents may naturally have lower ratios
```

### Issue: Very slow processing

**Solution**:
1. Reduce image size
2. Use faster filters:
   - Median instead of Bilateral
   - Standard Equalization instead of CLAHE
   - Otsu's instead of Adaptive
3. Close other applications

---

## Understanding the Metrics

### Quick Reference

| Metric | Poor | Fair | Good | Excellent |
|--------|------|------|------|-----------|
| PSNR (dB) | <20 | 20-30 | 30-35 | >35 |
| SSIM | <0.70 | 0.70-0.85 | 0.85-0.95 | >0.95 |
| MSE | >500 | 100-500 | 50-100 | <50 |
| Clarity | <100 | 100-300 | 300-500 | >500 |

### What Each Metric Means

**PSNR (Peak Signal-to-Noise Ratio)**
- **30 dB**: Good quality, slight differences visible
- **35 dB**: Excellent quality, very similar
- **40 dB+**: Near perfect

**SSIM (Structural Similarity)**
- **0.80**: Similar
- **0.90**: Very similar
- **0.95**: Nearly identical

**MSE (Mean Squared Error)**
- Lower is always better
- 0 means identical images

**Clarity (Laplacian Variance)**
- >500: Very sharp
- 200-500: Clear
- <100: Blurry

---

## Advanced Tips

### For Maximum Compression
1. Use Otsu's thresholding (more uniform regions)
2. Use RLE compression (perfect for binary)
3. Consider downsampling slightly

### For Maximum Clarity
1. Use Bilateral filter (edge preservation)
2. Use CLAHE (local contrast enhancement)
3. Use Adaptive thresholding

### For Fastest Processing
1. Use Gaussian filter (not Median or Bilateral)
2. Use standard Histogram Equalization
3. Use Otsu's thresholding
4. Use RLE compression

### For Best OCR Readiness
1. Clean with Median filter
2. Use CLAHE for enhancement
3. Use Otsu's thresholding
4. Save as PNG (lossless)

---

## Batch Processing (Manual)

To process multiple documents:

1. Run the app
2. Upload first image
3. Download results
4. Upload next image
5. Repeat

**Note**: App processes one document at a time

---

## Performance on Different Systems

### Small Image (500×500)
- Processing time: 1-2 seconds
- Memory: ~50 MB

### Medium Image (2000×2000)
- Processing time: 5-10 seconds
- Memory: ~200 MB

### Large Image (4000×4000)
- Processing time: 20-30 seconds
- Memory: ~500 MB

---

## System Requirements

**Minimum**:
- Python 3.8+
- 2 GB RAM
- 500 MB disk space
- Dual-core processor

**Recommended**:
- Python 3.10+
- 4 GB RAM
- 1 GB disk space
- Quad-core processor

---

## File Format Support

**Input Formats**:
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)

**Output Formats**:
- PNG (lossless, for processed images)
- TXT (for text report)

---

## Keyboard Shortcuts

- `Ctrl+C`: Stop the app
- `R`: Reload/rerun the app
- `Ctrl+Shift+R`: Hard reload

---

## Getting Help

### Check These First
1. README.md - Full documentation
2. TECHNICAL_DOCUMENTATION.md - Deep technical details
3. This Quick Start - Common issues
4. Code comments - Function documentation

### Additional Resources
- OpenCV documentation: https://docs.opencv.org/
- Streamlit documentation: https://docs.streamlit.io/
- scikit-image: https://scikit-image.org/

---

## Next Steps

After getting comfortable with the app:

1. **Experiment**: Try different parameter combinations
2. **Analyze**: Study the quality metrics
3. **Compare**: Compare different methods
4. **Extend**: Add your own filters or metrics
5. **Deploy**: Share the app with others

---

## Quick Parameter Cheat Sheet

```
MEDIAN FILTER:
3 = Light denoising, minimal blur
5 = Balanced (default, recommended)
7 = Heavy denoising, more blur
9+ = Very aggressive

GAUSSIAN FILTER:
sigma 0.5 = Sharp, preserves detail
sigma 1.0 = Balanced
sigma 2.0 = Heavy smoothing
sigma 3.0+ = Very smooth

CLAHE:
clip 1.0 = Conservative
clip 2.0 = Balanced (default)
clip 3.0 = Moderate enhancement
clip 4.0+ = Aggressive

ADAPTIVE THRESHOLD:
block 5 = Fine local detail
block 11 = Balanced (default)
block 21 = Coarse local regions
```

---

**Ready to enhance your documents!** 📄✨

For more details, see README.md
