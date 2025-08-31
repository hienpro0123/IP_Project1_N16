import numpy as np, matplotlib.pyplot as plt, cv2
def hist_equalization(img_gray, L=256):
    if img_gray.dtype == np.uint8:
        img_u8 = img_gray
    else:
        np.clip(img_gray,0,255).astype(np.uint8)
    hist = np.bincount(img_u8.ravel(), minlength=L)
    cdf  = np.cumsum(hist) / hist.sum()
    T    = np.round((L - 1) * cdf).astype(np.uint8)
    img_eq = T[img_u8]
    return img_eq, T, hist, cdf


def clahe_(img, clip=2.0, tile=8):
    g = img
    if g.dtype != np.uint8:
        g = (g*255.0 if g.max()<=1.0 else g).clip(0,255).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile,tile)).apply(g)


# --- Negative ---
def dao_anh(img):
    return 255 - img   # cho ảnh uint8

def img_process(img):
    neg = dao_anh(img)
    # HE & CLAHE trên ảnh negative
    img_he, T, hist, cdf = hist_equalization(neg)  # <-- unpack đúng
    img_clahe = clahe_(neg)
    return img, img_he,img_clahe


# --- Gamma ---
def chuyen_doi_Gamma(img, gamma, c):
    f = img.astype(np.float32) / 255.0           # [0,1]
    g = c * np.power(f, float(gamma))            # áp dụng gamma + hệ số c
    out = np.clip(g, 0.0, 1.0) * 255.0
    return out.astype(np.uint8)


def img_process(img):
    gamma = 2 #tuy chinh
    c = 1
    gm = chuyen_doi_Gamma(img, gamma, c)
    # HE & CLAHE trên ảnh 
    img_he, T, hist, cdf = hist_equalization(gm)  # <-- unpack đúng
    img_clahe = clahe_(gm)
    return img, img_he,img_clahe

#Contrast Stretching
def pixelVal(pix, r1, s1, r2, s2):
    if 0 <= pix <= r1:
        return (s1 / max(r1, 1e-6)) * pix
    elif r1 < pix <= r2:
        return ((s2 - s1) / max((r2 - r1), 1e-6)) * (pix - r1) + s1
    else:
        return ((255 - s2) / max((255 - r2), 1e-6)) * (pix - r2) + s2

def img_process(img):
    r1, s1, r2, s2 = 70, 10, 200, 255
    contrast_stretched = np.vectorize(pixelVal)(img, r1, s1, r2, s2)
    contrast_stretched = np.clip(contrast_stretched, 0, 255).astype(np.uint8)
    # HE & CLAHE trên ảnh 
    img_he, T, hist, cdf = hist_equalization(contrast_stretched)  # <-- unpack đúng
    img_clahe = clahe_(contrast_stretched)
    return img, img_he,img_clahe
