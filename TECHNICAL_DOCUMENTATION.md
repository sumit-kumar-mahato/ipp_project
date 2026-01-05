# Document Image Enhancement & Compression System
## Technical Documentation

---

## 📋 Table of Contents

1. [System Architecture](#system-architecture)
2. [Image Processing Pipeline](#image-processing-pipeline)
3. [Implementation Details](#implementation-details)
4. [Quality Metrics](#quality-metrics)
5. [Compression Algorithms](#compression-algorithms)
6. [API Reference](#api-reference)
7. [Performance Analysis](#performance-analysis)

---

## System Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Web Interface                   │
│              (5 Tabs: Preview, Processing, etc.)             │
└────┬────────────────────────────────────────────────────────┘
     │
     ├─► File Upload Handler
     │   └─ PIL/OpenCV Image Loading
     │
     ├─► Image Processing Pipeline
     │   ├─ Grayscale Conversion (OpenCV)
     │   ├─ Noise Removal Module
     │   ├─ Histogram Enhancement Module
     │   ├─ Binarization Module
     │   └─ Compression Module
     │
     ├─► Quality Metrics Calculator
     │   ├─ MSE Calculator
     │   ├─ PSNR Calculator
     │   ├─ SSIM Calculator
     │   └─ Clarity Scorer
     │
     └─► Visualization & Reporting
         ├─ Matplotlib Charts
         ├─ Statistical Tables
         └─ Download Manager
```

---

## Image Processing Pipeline

### Stage 1: Input & Grayscale Conversion

```python
def convert_to_grayscale(image):
    """Convert BGR (OpenCV) to grayscale using luminance weights"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

**Formula**: `Gray = 0.299*R + 0.587*G + 0.114*B`
- Uses perceptually weighted luminance
- Weights match human eye sensitivity
- Single channel reduces computation

**Why**: Document processing focuses on intensity patterns, not color

---

### Stage 2: Noise Removal

#### 2.1 Median Filter
```python
def apply_median_filter(image, kernel_size=5):
    return cv2.medianBlur(image, kernel_size)
```

**Algorithm**:
1. Slide kernel window over image
2. Extract pixel values in window
3. Sort values
4. Replace center pixel with median

**Mathematical Basis**:
- Non-linear filter
- Robust to outliers
- Preserves edges better than mean

**Best For**:
- Salt-and-pepper noise (impulse noise)
- Pixel spikes and artifacts
- Document scanning noise

**Kernel Sizes**:
- 3×3: Light filtering, minimal blur
- 5×5: Balanced (default)
- 7×7+: Heavy filtering, more blur

---

#### 2.2 Gaussian Filter
```python
def apply_gaussian_filter(image, kernel_size=5, sigma=1.0):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
```

**Algorithm**:
- Uses 2D Gaussian kernel
- Weights pixels by distance (bell curve)
- Separable convolution: O(n) instead of O(n²)

**Mathematical Formula**:
```
G(x,y) = (1/(2πσ²)) * exp(-(x²+y²)/(2σ²))
```

**Parameters**:
- Sigma: Standard deviation of Gaussian
  - Small σ: Sharp transitions preserved
  - Large σ: More smoothing

**Best For**:
- Gaussian noise (random fluctuations)
- Smooth gradual transitions needed

---

#### 2.3 Bilateral Filter
```python
def apply_bilateral_filter(image, diameter=9, 
                          sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(image, diameter, 
                              sigma_color, sigma_space)
```

**Algorithm**:
- Combines spatial and intensity Gaussians
- Preserves edges during smoothing
- Computationally expensive

**Formula**:
```
Output(x,y) = (1/W) * Σ I(xi,yi) * 
              exp(-||p-pi||²/(2σ_s²) - |I(p)-I(pi)|²/(2σ_r²))
```

**Parameters**:
- Diameter: Pixel neighborhood size
- Sigma Color: Range of intensity differences to consider
- Sigma Space: Range of spatial distances to consider

**Best For**:
- Preserving important edges
- Medical imaging (detail preservation)
- Complex document structures

---

### Stage 3: Histogram Enhancement

#### 3.1 Standard Histogram Equalization
```python
def apply_histogram_equalization(image_gray):
    return cv2.equalizeHist(image_gray)
```

**Algorithm**:
1. Compute histogram H
2. Compute cumulative distribution CDF
3. Normalize CDF to [0, 255]
4. Map pixels using CDF

**Mathematical Process**:
```
CDF(k) = (255 / N) * Σ H(i) for i=0 to k
Output = CDF[pixel_value]
```

**Effect**:
- Spreads pixel intensities evenly
- Increases contrast
- Can over-enhance noise in flat areas

**Best For**:
- Uniformly underexposed documents
- Clear foreground/background separation

---

#### 3.2 CLAHE (Contrast Limited Adaptive Histogram Equalization)
```python
def apply_clahe(image_gray, clip_limit=2.0, tile_size=8):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, 
                            tileGridSize=(tile_size, tile_size))
    return clahe.apply(image_gray)
```

**Algorithm**:
1. Divide image into tiles (e.g., 8×8)
2. Compute histogram equalization per tile
3. Apply clip limit to prevent noise amplification
4. Interpolate tile boundaries

**Parameters**:
- Clip Limit (1-5): Prevents over-enhancement
  - 1.0: Conservative
  - 2.0: Balanced
  - 4.0+: Aggressive
- Tile Size (4-16): Local region size
  - Smaller: More detail
  - Larger: More global

**Best For**:
- Uneven lighting conditions
- Shadow/highlight preservation
- Complex documents

---

### Stage 4: Binarization

#### 4.1 Otsu's Thresholding
```python
def apply_otsu_binarization(image_gray):
    _, binary = cv2.threshold(image_gray, 0, 255, 
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary
```

**Algorithm**:
1. Analyze all possible threshold values (0-255)
2. For each threshold T:
   - Compute class variance: σ²(T) = w0*w1*(μ0-μ1)²
   - w0, w1 = class weights
   - μ0, μ1 = class means
3. Select T that maximizes between-class variance

**Mathematical Foundation**:
```
σ²_between = w0(μ0 - μ)² + w1(μ1 - μ)²
Otsu_threshold = argmax(σ²_between)
```

**Characteristics**:
- Fully automatic (no manual threshold needed)
- Optimal for bimodal histograms
- Sensitive to lighting variations

**Best For**:
- Documents with clear text/background
- Uniform lighting conditions
- High contrast documents

---

#### 4.2 Adaptive Thresholding
```python
def apply_adaptive_threshold(image_gray, block_size=11):
    return cv2.adaptiveThreshold(image_gray, 255, 
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY, block_size, 2)
```

**Algorithm**:
1. For each pixel (x,y):
   - Compute mean of block_size×block_size neighborhood
   - Apply Gaussian weighting to neighborhood
   - Compare pixel to weighted mean
   - Threshold based on local comparison

**Formula**:
```
Output(x,y) = 255 if I(x,y) > (weighted_mean - constant)
              else 0
```

**Parameters**:
- Block Size: Neighborhood size for local mean
- Constant: Subtracted from weighted mean

**Best For**:
- Uneven lighting (shadows, glare)
- Documents with varying background
- Non-uniform document conditions

---

### Stage 5: Compression

#### 5.1 RLE (Run-Length Encoding)
```python
def rle_compress(data):
    """Encode consecutive identical bytes"""
    compressed = []
    current_byte = data[0]
    count = 1
    
    for i in range(1, len(data)):
        if data[i] == current_byte and count < 255:
            count += 1
        else:
            compressed.append(bytes([count, current_byte]))
            current_byte = data[i]
            count = 1
    
    compressed.append(bytes([count, current_byte]))
    return b''.join(compressed)
```

**Algorithm**:
- Identify runs of identical bytes
- Replace with (count, value) pairs
- Count limited to 255 (1 byte)

**Example**:
```
Input:  [255, 255, 255, 0, 0, 255]
Output: [(3, 255), (2, 0), (1, 255)]
Size:   6 bytes → 6 bytes (no improvement here)
```

**Compression Ratio**:
```
Ratio = ((Original - Compressed) / Original) * 100%
```

**Best For**:
- Binary images with large uniform regions
- Fax documents
- Simple diagrams
- High compression for text documents

**Performance**:
- Compression ratio for documents: 60-90%
- Speed: Very fast (linear O(n))
- Decompression: Very fast

---

#### 5.2 Huffman Compression (zlib)
```python
def huffman_compress(image_binary):
    """Compress using Huffman encoding (zlib)"""
    img_bytes = image_binary.tobytes()
    return zlib.compress(img_bytes, level=9)
```

**Algorithm** (Simplified):
1. Analyze byte frequency distribution
2. Build Huffman tree (frequent bytes → shorter codes)
3. Assign variable-length codes
4. Encode data using code table

**Compression Mechanism**:
- Frequent bytes: Fewer bits (e.g., 2-3 bits)
- Rare bytes: More bits (e.g., 8-10 bits)
- Average: Less than original 8 bits per byte

**zlib Implementation**:
- Uses DEFLATE (LZ77 + Huffman)
- Multi-pass compression
- Level 9: Maximum compression (slowest)

**Best For**:
- Complex pattern documents
- Mixed binary and grayscale
- Higher compression needed
- Less time-sensitive applications

---

## Quality Metrics

### 1. MSE (Mean Squared Error)
```python
def calculate_mse(img1, img2):
    return mean_squared_error(img1, img2)

# Formula: MSE = (1/N) * Σ(Original[i] - Enhanced[i])²
```

**Interpretation**:
- Range: 0 to ∞
- 0 = Identical images
- Higher = More different
- Sensitive to overall brightness shifts

**Limitations**:
- Pixel-by-pixel only
- Doesn't account for perception
- Shifted images show high MSE despite visual similarity

---

### 2. PSNR (Peak Signal-to-Noise Ratio)
```python
def calculate_psnr(img1, img2):
    mse = mean_squared_error(img1, img2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Formula: PSNR = 20*log10(255/√MSE)
```

**Scale**:
- < 20 dB: Low quality, visible degradation
- 20-30 dB: Fair quality
- 30-40 dB: Good quality
- > 40 dB: Excellent quality

**Advantages**:
- Logarithmic (more intuitive)
- Industry standard
- Easy to compare

**Limitations**:
- Doesn't measure perceptual quality
- Can be high even with visible artifacts

---

### 3. SSIM (Structural Similarity Index)
```python
def calculate_ssim(img1, img2):
    return structural_similarity(img1, img2, data_range=255)

# Formula: SSIM = (2μx*μy + C1)(2σxy + C2) / 
#                 [(μx² + μy² + C1)(σx² + σy² + C2)]
```

**Components**:
- Luminance (brightness) similarity
- Contrast similarity
- Structural similarity

**Scale**:
- -1 to 1 (typically 0 to 1)
- 1.0 = Perfect match
- 0.9+ = Very similar
- 0.8+ = Similar
- < 0.5 = Quite different

**Advantages**:
- Aligns with human perception
- Captures structural changes
- Not fooled by brightness shifts

---

### 4. Clarity Score (Laplacian Variance)
```python
def calculate_clarity_metrics(image_gray):
    laplacian_var = cv2.Laplacian(image_gray, cv2.CV_64F).var()
    return laplacian_var

# Uses 2nd derivative to detect edges
```

**How It Works**:
1. Apply Laplacian kernel (edge detector)
2. Compute variance of Laplacian response
3. High variance = Many edges = Sharp image
4. Low variance = Smooth = Blurry image

**Laplacian Kernel**:
```
[0  1  0]
[1 -4  1]
[0  1  0]
```

**Interpretation**:
- > 500: Very sharp
- 200-500: Good clarity
- 100-200: Moderate
- < 100: Blurry

---

## API Reference

### Noise Removal Functions

#### apply_median_filter(image, kernel_size=5)
- **Parameters**: 
  - image: Grayscale numpy array
  - kernel_size: Odd integer (3, 5, 7, 9, 11)
- **Returns**: Filtered image (numpy array)
- **Time Complexity**: O(n * k log k) where k = kernel_size
- **Space Complexity**: O(k²)

#### apply_gaussian_filter(image, kernel_size=5, sigma=1.0)
- **Parameters**:
  - image: Grayscale numpy array
  - kernel_size: Odd integer
  - sigma: Float (standard deviation)
- **Returns**: Filtered image
- **Time Complexity**: O(n * k)
- **Space Complexity**: O(k)

#### apply_bilateral_filter(image, diameter=9, sigma_color=75, sigma_space=75)
- **Parameters**:
  - image: Grayscale numpy array
  - diameter: Neighborhood diameter
  - sigma_color: Range of color space
  - sigma_space: Range of spatial space
- **Returns**: Filtered image
- **Time Complexity**: O(n * d²) where d = diameter
- **Note**: Most computationally expensive

---

### Enhancement Functions

#### apply_histogram_equalization(image_gray)
- **Returns**: Enhanced image with spread histogram
- **Time Complexity**: O(n)
- **Effect**: Stretches intensities to full range

#### apply_clahe(image_gray, clip_limit=2.0, tile_size=8)
- **Returns**: Locally enhanced image
- **Time Complexity**: O(n)
- **Effect**: Adaptive local enhancement

---

### Binarization Functions

#### apply_otsu_binarization(image_gray)
- **Returns**: Binary image (0 or 255 only)
- **Time Complexity**: O(256*n) = O(n)
- **Auto-selects** optimal threshold

#### apply_adaptive_threshold(image_gray, block_size=11)
- **Parameters**:
  - block_size: Must be odd integer
- **Returns**: Binary image
- **Time Complexity**: O(n * block_size²)

---

### Compression Functions

#### rle_compress(data)
- **Input**: Byte array
- **Returns**: Compressed byte array
- **Time Complexity**: O(n)
- **Space**: Best case O(n/k), worst case O(n)

#### huffman_compress(image_binary)
- **Input**: Grayscale numpy array
- **Returns**: Compressed byte array
- **Time Complexity**: O(n log n)
- **Note**: Uses zlib library internally

#### calculate_compression_ratio(original_size, compressed_size)
- **Returns**: Percentage reduction
- **Formula**: ((Original - Compressed) / Original) * 100

---

### Quality Metrics

#### calculate_mse(img1, img2)
- **Time Complexity**: O(n)
- **Returns**: Float (MSE value)

#### calculate_psnr(img1, img2)
- **Time Complexity**: O(n)
- **Returns**: Float (dB value)

#### calculate_ssim(img1, img2)
- **Time Complexity**: O(n)
- **Returns**: Float (-1 to 1)

#### calculate_clarity_metrics(image_gray)
- **Time Complexity**: O(n)
- **Returns**: Float (Laplacian variance)

---

## Performance Analysis

### Computational Complexity

| Operation | Time | Space | Notes |
|-----------|------|-------|-------|
| Grayscale Conversion | O(n) | O(n) | Linear pixel access |
| Median Filter (k×k) | O(n*k log k) | O(k²) | Sorting in window |
| Gaussian Filter | O(n*k) | O(k) | Separable convolution |
| Bilateral Filter | O(n*d²) | O(d²) | Most expensive |
| Histogram Equalization | O(n) | O(256) | Single pass |
| CLAHE | O(n) | O(tile_size²) | Local computation |
| Otsu Thresholding | O(n) | O(256) | All thresholds tested |
| Adaptive Threshold | O(n*block²) | O(block²) | Per-pixel local mean |
| RLE Compression | O(n) | O(n) | Linear scan |
| Huffman Compression | O(n log n) | O(n) | zlib complexity |
| MSE Calculation | O(n) | O(1) | Single pass difference |
| SSIM Calculation | O(n) | O(1) | Sliding window statistics |

### Memory Usage

For 1000×1000 image:
- Original: 1 MB
- Processing: 2-3 MB (working arrays)
- Compressed (RLE): 50-200 KB
- Compressed (Huffman): 30-150 KB

### Processing Time

Typical values on modern CPU:

| Operation | Time (1000×1000) |
|-----------|-----------------|
| Grayscale | < 1 ms |
| Median Filter (5×5) | 50-100 ms |
| Gaussian Filter | 10-20 ms |
| Bilateral Filter | 100-500 ms |
| Histogram Equalization | < 5 ms |
| CLAHE | 20-50 ms |
| Binarization | < 5 ms |
| Compression (RLE) | < 10 ms |
| Compression (Huffman) | 50-200 ms |

---

## Optimization Strategies

### Memory
- Process in blocks for very large images
- Use uint8 arrays (not float64)
- Release intermediate arrays

### Speed
1. Use Gaussian Filter (not Median) for speed
2. Use standard equalization (not CLAHE)
3. Use Otsu's thresholding
4. Use RLE for compression

### Quality
1. Use Bilateral Filter (preserves edges)
2. Use CLAHE (local enhancement)
3. Use Adaptive Thresholding
4. Use Huffman compression

---

## Testing Recommendations

### Unit Tests
```python
def test_noise_removal():
    # Add known noise, verify removal
    
def test_histogram_equalization():
    # Check histogram spread
    
def test_compression_ratio():
    # Verify ratio calculation
```

### Integration Tests
```python
def test_full_pipeline():
    # Test complete processing flow
    # Verify output quality
```

---

## References

1. **Median Filtering**: Gonzalez & Woods, Digital Image Processing
2. **Bilateral Filtering**: Tomasi & Manduchi (1998)
3. **CLAHE**: Zuiderveld, K. (1994)
4. **Otsu Thresholding**: Otsu, N. (1979)
5. **Structural Similarity**: Wang et al. (2004)
6. **RLE**: Classic compression technique
7. **Huffman Coding**: Huffman, D. (1952)

---

**Document Version**: 1.0
**Last Updated**: 2026-01-04
**Status**: Production Ready