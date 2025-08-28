import numpy as np, matplotlib.pyplot as plt, cv2

# (Gray-Level Slicing)
def slicedGreyScale(image,T1=150, T2=240, keep_bg=True):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        mask = (gray > T1) & (gray < T2)
        out = gray.copy() if keep_bg else np.zeros_like(gray, np.uint8)
        out[mask] = np.clip(gray[mask].astype(np.int16) + 60, 0, 255).astype(np.uint8)
        return out


# --- HE tự viết cho ảnh xám uint8 ---
def equalizeHistogram(img):
    N = img.size
    hist = np.bincount(img.ravel(), minlength=256)
    nz = np.flatnonzero(hist)
    if nz.size == 0: return img.copy()
    cdf = hist.cumsum()
    cdf_min = cdf[nz[0]]
    if cdf_min == N: return img.copy()
    lut = np.floor((cdf - cdf_min) * 255.0 / (N - cdf_min)).clip(0, 255).astype(np.uint8)
    return lut[img]


# --- Đọc ảnh xám---
filename ="img13.jpg"
img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)

res = slicedGreyScale(img, T1=160, T2=250, keep_bg=True)

# HE & CLAHE trên ảnh negative
img_he = equalizeHistogram(res)
clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8,8))
img_clahe = clahe.apply(res)

# --- Hiển thị 4 ảnh + 4 histogram ---
fig, axs = plt.subplots(
    2, 2,
    figsize=(18, 10),     # to hơn
    dpi=150,              # nét hơn
    layout="constrained"  # tự canh sát nhau
)

axs[0,0].imshow(img, cmap='gray', vmin=0, vmax=255);  axs[0,0].set_title('Original');           axs[0,0].axis('off')
axs[0,1].imshow(res, cmap='gray', vmin=0, vmax=255);  axs[0,1].set_title('Gray-Level Slicing');           axs[0,1].axis('off')
axs[1,0].imshow(img_he, cmap='gray', vmin=0, vmax=255); axs[1,0].set_title('HE'); axs[1,0].axis('off')
axs[1,1].imshow(img_clahe, cmap='gray', vmin=0, vmax=255); axs[1,1].set_title('CLAHE'); axs[1,1].axis('off')


