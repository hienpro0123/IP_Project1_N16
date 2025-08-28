import numpy as np, matplotlib.pyplot as plt, cv2

#Contrast Stretching
def pixelVal(pix, r1, s1, r2, s2):
    if 0 <= pix <= r1:
        return (s1 / max(r1, 1e-6)) * pix
    elif r1 < pix <= r2:
        return ((s2 - s1) / max((r2 - r1), 1e-6)) * (pix - r1) + s1
    else:
        return ((255 - s2) / max((255 - r2), 1e-6)) * (pix - r2) + s2


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


# --- Đọc ảnh xám (nếu ảnh màu muốn giữ màu, xem ghi chú bên dưới) ---
filename = r"img7.jpg"
img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)

r1, s1, r2, s2 = 70, 10, 200, 255
contrast_stretched = np.vectorize(pixelVal)(img, r1, s1, r2, s2)
contrast_stretched = np.clip(contrast_stretched, 0, 255).astype(np.uint8)

# HE & CLAHE trên ảnh negative
img_he = equalizeHistogram(contrast_stretched)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
img_clahe = clahe.apply(contrast_stretched)

# --- Hiển thị 4 ảnh + 4 histogram ---
fig, axs = plt.subplots(
    2, 2,
    figsize=(18, 10),     # to hơn
    dpi=150,              # nét hơn
    layout="constrained"  # tự canh sát nhau
)

axs[0,0].imshow(img, cmap='gray', vmin=0, vmax=255);  axs[0,0].set_title('Original');           axs[0,0].axis('off')
axs[0,1].imshow(contrast_stretched, cmap='gray', vmin=0, vmax=255);  axs[0,1].set_title('Contrast Stretching');           axs[0,1].axis('off')
axs[1,0].imshow(img_he, cmap='gray', vmin=0, vmax=255); axs[1,0].set_title('HE'); axs[1,0].axis('off')
axs[1,1].imshow(img_clahe, cmap='gray', vmin=0, vmax=255); axs[1,1].set_title('CLAHE'); axs[1,1].axis('off')


