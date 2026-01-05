# Document Image Enhancement & Compression System

A comprehensive **Streamlit web application** for cleaning, enhancing, and compressing scanned document images using advanced image processing techniques.

---

## 🎯 Project Overview

This application implements a complete **document digitization pipeline** that:

1. **Removes noise** from scanned documents using median, Gaussian, or bilateral filters
2. **Enhances clarity** through histogram equalization or CLAHE
3. **Binarizes images** using Otsu's or adaptive thresholding
4. **Compresses efficiently** using RLE and Huffman encoding
5. **Analyzes quality** with PSNR, SSIM, MSE, and clarity metrics

**Real-world Applications:**
- Office automation systems
- Digital document archiving
- Medical imaging workflows
- OCR preprocessing pipelines
- Archival document digitization

---

## 📦 Features

### 1. **Noise Removal (Session 14 & Median Filtering)**
- **Median Filter**: Best for salt-and-pepper noise. Replaces each pixel with the median value in its neighborhood (3×3 to 11×11 kernels)
- **Gaussian Filter**: Best for Gaussian noise. Uses bell-curve weighting with tunable sigma
- **Bilateral Filter**: Edge-preserving smoothing. Maintains sharp boundaries while reducing noise

### 2. **Histogram Enhancement (Session 11 & 12)**
- **Standard Histogram Equalization**: Stretches pixel intensity values evenly across 0-255 range
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization): Prevents noise amplification through clip limit
- **Real-time histogram visualization**: Compare before/after distributions

### 3. **Binarization (Session 15 - Object Detection)**
- **Otsu's Thresholding**: Automatic optimal threshold by minimizing intra-class variance
- **Adaptive Thresholding**: Region-based thresholding for uneven lighting
- **Threshold value display**: Shows automatically selected threshold

### 4. **Compression Techniques**
- **RLE (Run-Length Encoding)**: Encodes consecutive identical bytes as (count, value) pairs. Excellent for binary images with large uniform regions
- **Huffman Compression (zlib)**: Assigns variable-length codes based on frequency distribution
- **Compression ratio comparison**: Visual and statistical comparison

### 5. **Quality Metrics (Session 3 - Quality Assessment)**
- **MSE** (Mean Squared Error): Pixel-wise difference metric
- **PSNR** (Peak Signal-to-Noise Ratio): Signal-to-noise ratio in decibels
- **SSIM** (Structural Similarity Index): Perceptual similarity (0-1 scale)
- **Clarity Score**: Laplacian variance measure of sharpness

### 6. **Statistical Report**
- Detailed processing pipeline summary
- Compression effectiveness analysis
- Quality metric interpretation
- Actionable recommendations

---

## 🏗️ Architecture & Pipeline

```
Input Image (Scanned Document)
    ↓
1. GRAYSCALE CONVERSION (Session 3)
    ↓
2. NOISE REMOVAL (Session 14)
   ├─ Median Filter (salt-and-pepper)
   ├─ Gaussian Filter (Gaussian noise)
   └─ Bilateral Filter (edge-preserving)
    ↓
3. HISTOGRAM ENHANCEMENT (Session 11-12)
   ├─ Histogram Equalization
   └─ CLAHE (Contrast Limited)
    ↓
4. BINARIZATION (Session 15)
   ├─ Otsu's Thresholding (automatic)
   └─ Adaptive Thresholding (region-based)
    ↓
5. COMPRESSION (Session 3 - File Formats)
   ├─ RLE Compression
   └─ Huffman Compression (zlib)
    ↓
6. QUALITY ANALYSIS (Session 3)
   ├─ MSE, PSNR, SSIM Calculation
   ├─ Clarity Metrics
   └─ Statistical Report
    ↓
Output (Enhanced + Compressed Documents)
```

---

## 📊 Image Processing Concepts Used

### From Session 1-2 (Fundamentals)
- Pixel concept and image as data matrix
- Grayscale conversion using luminance weights
- RGB to grayscale transformation

### From Session 3-4 (Properties & Quality)
- Image resolution and aspect ratio
- Quality metrics: MSE, PSNR, SSIM
- Lossless vs Lossy compression

### From Session 5-6 (Transformations)
- Image cropping for ROI selection
- Coordinate systems understanding

### From Session 11-12 (Histograms & Filtering)
- Histogram computation and visualization
- Histogram equalization for contrast enhancement
- Low-pass filters (Gaussian) for smoothing
- Laplacian filter for edge detection/clarity

### From Session 13-14 (Noise & Sharpening)
- Image noise types: Gaussian, Salt-and-pepper, Speckle
- Median filtering for impulse noise removal
- Bilateral filtering for edge preservation
- Noise impact on downstream applications

### From Session 15-16 (Object Detection)
- Otsu's thresholding methodology
- Adaptive thresholding for varying lighting
- Binary segmentation concept
- Clarity metrics using Laplacian variance

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Step 1: Clone/Download the Project
```bash
# Download and navigate to project directory
cd document-enhancement-compression
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
streamlit run document_app.py
```

### Step 5: Access in Browser
The app will automatically open at:
```
http://localhost:8501
```

---

## 📋 Requirements

See `requirements.txt`:

```
streamlit>=1.28.0
opencv-python>=4.8.0
Pillow>=10.0.0
matplotlib>=3.7.0
numpy>=1.24.0
scikit-image>=0.21.0
```

---

## 🎮 Usage Guide

### Basic Workflow

1. **Upload Document**
   - Click "Upload Document Image"
   - Select JPG, PNG, BMP, or TIFF file
   - Maximum recommended: 10 MB

2. **Configure Parameters (Sidebar)**
   - **Noise Removal**: Choose method and adjust kernel size
   - **Enhancement**: Select histogram equalization or CLAHE
   - **Binarization**: Choose Otsu or Adaptive thresholding
   - **Compression**: Select RLE, Huffman, or both

3. **Preview Tab**
   - View original image dimensions
   - Check original file size
   - Understand image characteristics

4. **Processing Tab**
   - See noise removal results
   - Compare histogram before/after
   - Observe clarity improvements

5. **Binarization Tab**
   - View binary conversion
   - Check foreground/background ratio
   - See automatically selected threshold

6. **Compression Tab**
   - Compare compression ratios
   - View file size reduction
   - Analyze compression efficiency

7. **Analysis & Report Tab**
   - Review quality metrics
   - Read recommendations
   - Download processed images and reports

---

## 📈 Example Scenarios

### Scenario 1: Old Scanned Document
```
Problem: Faded, uneven lighting, Gaussian noise
Solution:
1. Noise Removal: Gaussian Filter (smooths noise)
2. Enhancement: CLAHE (handles uneven lighting)
3. Binarization: Adaptive Thresholding (region-based)
4. Compression: Both RLE & Huffman
Result: Clean, readable, compressed document
```

### Scenario 2: Receipt Image
```
Problem: Salt-and-pepper noise, low contrast
Solution:
1. Noise Removal: Median Filter (removes impulse noise)
2. Enhancement: Histogram Equalization (improves contrast)
3. Binarization: Otsu's Method (automatic threshold)
4. Compression: RLE (high compression for binary)
Result: Clear, compact receipt image
```

### Scenario 3: Medical Document
```
Problem: Complex texture, subtle details, file size
Solution:
1. Noise Removal: Bilateral Filter (preserves edges)
2. Enhancement: CLAHE (maintains local contrasts)
3. Binarization: Adaptive Thresholding (uneven lighting)
4. Compression: Huffman (handles complexity)
Result: Detail-preserving, compressed medical image
```

---

## 📊 Understanding Metrics

### MSE (Mean Squared Error)
- **Range**: 0 to infinity
- **Interpretation**: Lower is better
- **When to use**: Pixel-level quality comparison
- **Limitation**: Doesn't account for perceptual quality

### PSNR (Peak Signal-to-Noise Ratio)
- **Range**: 0 to 100+ dB
- **Good Quality**: > 30 dB
- **Excellent Quality**: > 35 dB
- **Interpretation**: Higher is better
- **Advantage**: Standard in industry

### SSIM (Structural Similarity Index)
- **Range**: -1 to 1 (typically 0 to 1)
- **Perfect Match**: 1.0
- **Poor Quality**: < 0.5
- **Interpretation**: Closer to 1 is better
- **Advantage**: Aligns with human perception

### Clarity Score (Laplacian Variance)
- **Range**: 0 to very high numbers
- **Sharp Image**: High variance
- **Blurry Image**: Low variance
- **Interpretation**: Higher = Clearer
- **Use Case**: Focus detection

---

## 🔧 Advanced Configuration

### Noise Removal Parameters

**Median Filter**
```
Kernel Size: 3-11 (odd numbers)
- 3: Light noise removal, minimal blurring
- 5: Balanced (default)
- 7+: Heavy noise removal, more blurring
Best for: Salt-and-pepper noise
```

**Gaussian Filter**
```
Kernel Size: 3-11
Sigma: 0.5-5.0
- Low Sigma: Preserves details
- High Sigma: More smoothing
Best for: Gaussian noise
```

**Bilateral Filter**
```
Diameter: 5-15
Sigma Color: 10-150 (larger = more colors merged)
Sigma Space: 10-150 (larger = more spatial influence)
Best for: Edge-preserving smoothing
```

### CLAHE Parameters

```
Clip Limit: 1.0-5.0
- 1.0: Conservative enhancement
- 2.0-3.0: Balanced (default)
- 4.0+: Aggressive enhancement

Tile Size: 4-16
- Smaller: More local detail
- Larger: More global smoothing
```

---

## 📈 Performance Optimization

### For Large Documents (> 5 MB)
1. Use Median Filter (fastest noise removal)
2. Use standard Histogram Equalization (not CLAHE)
3. Use Otsu's thresholding (faster than adaptive)
4. Prefer RLE for binary compression

### For Quality Priority
1. Use Bilateral Filter (preserves edges)
2. Use CLAHE (better contrast)
3. Use Adaptive Thresholding (handles lighting)
4. Use Huffman compression (better for complex)

---

## 🐛 Troubleshooting

### Issue: App Crashes on Large Images
**Solution**: Resize image to < 5000×5000 pixels in an image editor first

### Issue: Poor Binarization Results
**Solution**: Try Adaptive Thresholding instead of Otsu's or vice versa

### Issue: Over-Denoising Causing Blurriness
**Solution**: Reduce kernel size or switch to Bilateral Filter

### Issue: Compression Ratio Too Low
**Solution**: Ensure you're working with truly binary images (black & white only)

---

## 📚 References from PPT Sessions

1. **Session 3-4**: Image Properties, Quality Metrics (MSE, PSNR, SSIM)
2. **Session 11-12**: Histograms, Histogram Equalization, Image Filtering
3. **Session 13-14**: Image Noise Types, Median & Bilateral Filters, Denoising
4. **Session 15-16**: Thresholding (Otsu, Adaptive), Binarization, Object Detection

---

## 💾 Output Files

The application generates:

1. **Enhanced Image** (PNG format)
   - Denoised and contrast-enhanced version
   - Full resolution preserved

2. **Binary Image** (PNG format)
   - Black and white output
   - Ideal for compression
   - Suitable for OCR

3. **Text Report** (TXT format)
   - Complete processing parameters
   - Quality metrics
   - Compression statistics
   - Recommendations

---

## 🎓 Educational Value

This project demonstrates:

✓ Image processing pipeline design
✓ Noise removal techniques
✓ Histogram-based enhancement
✓ Thresholding algorithms
✓ Compression methodologies
✓ Quality metrics
✓ Full-stack web application
✓ Data visualization
✓ Statistical analysis
✓ Real-world application design

---

## 👨‍💻 Code Organization

```
document_app.py
├─ Configuration & Setup
├─ Noise Removal Functions
├─ Histogram Equalization
├─ Binarization Methods
├─ Compression Functions
├─ Quality Metrics Calculation
├─ Utility Functions
└─ Streamlit UI (5 Tabs)
    ├─ Preview
    ├─ Processing
    ├─ Binarization
    ├─ Compression
    └─ Analysis & Report
```

---

## 🤝 Contributing

Feel free to enhance this project:
- Add more filters (Morphological operations)
- Implement edge detection
- Add OCR integration
- Support batch processing
- Add more compression algorithms

---

## 📄 License

Educational project for learning image processing concepts.

---

## ✨ Key Takeaways

1. **Preprocessing is Critical**: Noise removal and enhancement directly impact downstream tasks
2. **Method Selection Matters**: Different images need different techniques
3. **Trade-offs Exist**: Compression vs Quality, Speed vs Quality
4. **Metrics Guide Decisions**: Use PSNR, SSIM for quality assurance
5. **Real-world Complexity**: Documents have varied lighting, noise types, and requirements

---

**Built with:** Streamlit, OpenCV, Pillow, scikit-image
**Concepts from:** IPP Sessions 1-16 (Image Processing & Python)

Happy Processing! 📄✨
