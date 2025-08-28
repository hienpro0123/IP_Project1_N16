import numpy as np, matplotlib.pyplot as plt, cv2

# --- Logarit ---
def chuyen_doi_logarit(img):
    r = img.astype(np.float32)
    
    #np.log1p(x) = ln(1+x)
    
    # Tính c để kết quả về 0..255
    c = 255.0 / np.log1p(r.max())

    out = c * np.log1p(r)                    
    return np.clip(out, 0, 255).astype(np.uint8) #ep kieu ve 8 bit

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
filename = r"img2.jpg"
img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
assert img is not None, "Không đọc được ảnh"

# Logarit
log = chuyen_doi_logarit(img)

# HE & CLAHE trên ảnh negative
img_he = equalizeHistogram(log)
lahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_clahe = clahe.apply(log)

# --- Hiển thị 4 ảnh + 4 histogram ---
fig, axs = plt.subplots(
    2, 2,
    figsize=(18, 10),     # to hơn
    dpi=150,              # nét hơn
    layout="constrained"  # tự canh sát nhau
)

axs[0,0].imshow(img, cmap='gray', vmin=0, vmax=255);  axs[0,0].set_title('Original');           axs[0,0].axis('off')
axs[0,1].imshow(log, cmap='gray', vmin=0, vmax=255);  axs[0,1].set_title('Logarit');           axs[0,1].axis('off')
axs[1,0].imshow(img_he, cmap='gray', vmin=0, vmax=255); axs[1,0].set_title('HE (on Logarit)'); axs[1,0].axis('off')
axs[1,1].imshow(img_clahe, cmap='gray', vmin=0, vmax=255); axs[1,1].set_title('CLAHE (on Logarit)'); axs[1,1].axis('off')


