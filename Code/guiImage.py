import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2

APP_TITLE = "Tk + OpenCV Image Processing"
VIEW_W, VIEW_H = 520, 390  # kích thước khung preview mỗi ảnh

class ImageApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1160x600")
        self.minsize(900, 520)

        # Lưu/hiển thị ở BGR; xử lý trên ảnh xám
        self.img_bgr_orig = None
        self.img_bgr_work = None
        self.tk_img_left = None
        self.tk_img_right = None

        self._build_ui()
        self._bind_events()

    # ---------------- UI ----------------
    def _build_ui(self):
        bar = ttk.Frame(self)
        bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(bar, text="Open", command=self.on_open).pack(side=tk.LEFT, padx=4, pady=4)
        ttk.Button(bar, text="Save Result", command=self.on_save).pack(side=tk.LEFT, padx=4, pady=4)
        ttk.Button(bar, text="Reset", command=self.on_reset).pack(side=tk.LEFT, padx=4, pady=4)

        body = ttk.Frame(self); body.pack(fill=tk.BOTH, expand=True)

        # Previews
        left = ttk.Frame(body); left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 4), pady=8)
        previews = tk.PanedWindow(left, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        previews.pack(fill=tk.BOTH, expand=True)
        frm_left = ttk.Frame(previews); frm_right = ttk.Frame(previews)
        previews.add(frm_left, stretch="always"); previews.add(frm_right, stretch="always")
        self.panel_orig = self._make_view(frm_left, title="Original (Gray)")
        self.panel_proc = self._make_view(frm_right, title="Processed (Gray)")

        # Controls
        right = ttk.Frame(body); right.pack(side=tk.RIGHT, fill=tk.Y, padx=(4, 8), pady=8)
        ttk.Label(right, text="Operation:", font=("", 10, "bold")).pack(anchor="w", pady=(0, 4))
        self.op_var = tk.StringVar(value="None")
        ops = [
            "None",
            "Grayscale",
            "Negative",
            "Histogram Equalization",
            "CLAHE",
            "Gamma",
            "Logarithm",
            "Contrast Stretching",
            "Gray-Level Slicing",
        ]
        self.cbo_ops = ttk.Combobox(right, values=ops, textvariable=self.op_var, state="readonly")
        self.cbo_ops.pack(fill=tk.X)

        ttk.Separator(right).pack(fill=tk.X, pady=8)

        # Param stack
        self.param_stack = ttk.Frame(right); self.param_stack.pack(fill=tk.X)

        self.frm_none = ttk.Frame(self.param_stack)
        ttk.Label(self.frm_none, text="Không có tham số cho phép này").pack(anchor="w")

        # Gamma
        self.frm_gamma = ttk.Frame(self.param_stack)
        self.gamma_var = tk.DoubleVar(value=1.00)
        self._make_slider(self.frm_gamma, "Gamma", self.gamma_var, 0.10, 3.00, 0.01)

        # CLAHE
        self.frm_clahe = ttk.Frame(self.param_stack)
        self.clahe_clip = tk.DoubleVar(value=2.0)
        self._make_slider(self.frm_clahe, "clipLimit", self.clahe_clip, 1.0, 5.0, 0.1)
        self.clahe_tile = tk.IntVar(value=8)
        self._make_slider(self.frm_clahe, "tileGridSize", self.clahe_tile, 4, 32, 1, integer=True)

        # Contrast Stretching sliders (r1,s1,r2,s2)
        self.frm_cs = ttk.Frame(self.param_stack)
        self.cs_r1 = tk.IntVar(value=70);   self._make_slider(self.frm_cs, "r1", self.cs_r1, 0, 255, 1, integer=True)
        self.cs_s1 = tk.IntVar(value=30);   self._make_slider(self.frm_cs, "s1", self.cs_s1, 0, 255, 1, integer=True)
        self.cs_r2 = tk.IntVar(value=180);  self._make_slider(self.frm_cs, "r2", self.cs_r2, 0, 255, 1, integer=True)
        self.cs_s2 = tk.IntVar(value=220);  self._make_slider(self.frm_cs, "s2", self.cs_s2, 0, 255, 1, integer=True)

        # Gray-Level Slicing sliders (T1,T2) + checkbox keep_bg
        self.frm_slice = ttk.Frame(self.param_stack)
        self.sl_t1 = tk.IntVar(value=150);  self._make_slider(self.frm_slice, "T1", self.sl_t1, 0, 255, 1, integer=True)
        self.sl_t2 = tk.IntVar(value=240);  self._make_slider(self.frm_slice, "T2", self.sl_t2, 0, 255, 1, integer=True)
        self.keep_bg = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.frm_slice, text="Keep background", variable=self.keep_bg,
                        command=self.update_processed).pack(anchor="w", pady=(6, 0))

        # Default visible params
        self._show_params_for("None")

        # --- Combine (optional) ---
        combosect = ttk.LabelFrame(right, text="Combine (optional)")
        combosect.pack(fill=tk.X, pady=(6, 8))

        ttk.Label(combosect, text="Enhance with:").pack(anchor="w")
        self.enhance_var = tk.StringVar(value="None")
        self.cbo_enhance = ttk.Combobox(
            combosect, values=["None", "HE", "CLAHE"],
            textvariable=self.enhance_var, state="readonly"
        )
        self.cbo_enhance.pack(fill=tk.X, pady=2)

        self.combine_order = tk.StringVar(value="Enhance→Base")
        ttk.Radiobutton(
            combosect, text="Enhance → Base",
            variable=self.combine_order, value="Enhance→Base"
        ).pack(anchor="w")
        ttk.Radiobutton(
            combosect, text="Base → Enhance",
            variable=self.combine_order, value="Base→Enhance"
        ).pack(anchor="w")

        self.neg_after_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            combosect, text="Apply Negative at end",
            variable=self.neg_after_var
        ).pack(anchor="w", pady=(4, 0))

        # --- Chain extra ops (fixed order) ---
        chainsect = ttk.LabelFrame(right, text="Apply also (chain in order)")
        chainsect.pack(fill=tk.X, pady=(0, 8))

        self.ch_log   = tk.BooleanVar(value=False)
        self.ch_gam   = tk.BooleanVar(value=False)
        self.ch_cs    = tk.BooleanVar(value=False)
        self.ch_slice = tk.BooleanVar(value=False)
        self.ch_neg   = tk.BooleanVar(value=False)

        def on_chain_change():
            self.update_processed()
            self._show_params_for(self.op_var.get())  # hiện slider khi bật chain

        for text, var in [
            ("Logarithm", self.ch_log),
            ("Gamma", self.ch_gam),
            ("Contrast Stretching", self.ch_cs),
            ("Gray-Level Slicing", self.ch_slice),
            ("Negative", self.ch_neg),
        ]:
            ttk.Checkbutton(chainsect, text=text, variable=var,
                            command=on_chain_change).pack(anchor="w")

        ttk.Label(chainsect,
                  text="Order: Log → Gamma → ContrastStretch → Slice → Negative",
                  foreground="#666").pack(anchor="w", pady=(4, 0))

        ttk.Separator(right).pack(fill=tk.X, pady=8)
        ttk.Button(right, text="Apply / Update", command=self.update_processed).pack(fill=tk.X)
        ttk.Label(
            right,
            text=("\nMẹo:\n- HE/CLAHE: cải thiện vùng tối/sáng cục bộ"
                  "\n- Gamma < 1: sáng hơn; > 1: tối hơn"
                  "\n- Log: làm rõ vùng tối"),
            justify=tk.LEFT, foreground="#555"
        ).pack(fill=tk.X, pady=8)

        # bind thêm để cập nhật UI khi đổi enhance/order
        self.cbo_enhance.bind("<<ComboboxSelected>>",
                              lambda e: (self.update_processed(), self._show_params_for(self.op_var.get())))
        for rb in combosect.winfo_children():
            if isinstance(rb, ttk.Radiobutton):
                rb.configure(command=lambda: (self.update_processed(), self._show_params_for(self.op_var.get())))
        self._show_placeholder()

    def _make_view(self, parent, title=""):
        frm = ttk.LabelFrame(parent, text=title)
        frm.pack(fill=tk.BOTH, expand=True, pady=4)
        canvas = tk.Canvas(frm, width=VIEW_W, height=VIEW_H, bg="#222", highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        return canvas

    def _make_slider(self, parent, text, var, frm, to, res, integer=False):
        row = ttk.Frame(parent); row.pack(fill=tk.X, pady=4)
        ttk.Label(row, text=text).pack(side=tk.LEFT)
        val_lab = ttk.Label(row, text=f"{int(var.get())}" if integer else f"{var.get():.2f}")
        val_lab.pack(side=tk.RIGHT)
        scale = ttk.Scale(parent, from_=frm, to=to, variable=var,
                          command=lambda e: self._on_slider(var, val_lab, integer), orient=tk.HORIZONTAL)
        scale.pack(fill=tk.X)
        scale.bind("<ButtonRelease-1>", lambda e: self._snap_slider(var, res, integer))

    def _on_slider(self, var, lab, integer):
        v = var.get()
        lab.config(text=f"{int(round(v))}" if integer else f"{v:.2f}")

    def _snap_slider(self, var, step, integer):
        v = var.get()
        var.set(int(round(v / step) * step) if integer else round(v / step) * step)
        self.update_processed()

    def _bind_events(self):
        self.cbo_ops.bind("<<ComboboxSelected>>",
                          lambda e: (self._show_params_for(self.op_var.get()), self.update_processed()))
        self.bind("<Configure>", lambda e: self._redraw())

    # -------------- Actions --------------
    def on_open(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[("Image", ".png .jpg .jpeg .bmp .tif .tiff"), ("All", "*.*")]
        )
        if not path: return
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            messagebox.showerror("Error", "Không đọc được ảnh.")
            return
        self.img_bgr_orig = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self.img_bgr_work = self.img_bgr_orig.copy()
        self.update_previews()

    def on_save(self):
        if self.img_bgr_work is None:
            messagebox.showinfo("Info", "Chưa có ảnh để lưu.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", ".png"), ("JPEG", ".jpg .jpeg"), ("BMP", ".bmp")]
        )
        if not path: return
        if not cv2.imwrite(path, self.img_bgr_work):
            messagebox.showerror("Error", "Lưu thất bại.")

    def on_reset(self):
        if self.img_bgr_orig is not None:
            self.img_bgr_work = self.img_bgr_orig.copy()
            self.op_var.set("None")
            self._show_params_for("None")
            self.update_previews()

    def _show_placeholder(self):
        ph = np.full((VIEW_H, VIEW_W, 3), (40, 40, 40), np.uint8)
        cv2.putText(ph, "Open image...", (30, VIEW_H // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (180, 180, 180), 2, cv2.LINE_AA)
        self._draw_on_canvas(self.panel_orig, ph)
        self._draw_on_canvas(self.panel_proc, ph)

    def _show_params_for(self, op_name: str):
        # 1) Ẩn hết
        for child in self.param_stack.winfo_children():
            child.pack_forget()

        # 2) Khung theo base op
        mapping = {
            "None": self.frm_none,
            "Grayscale": self.frm_none,
            "Negative": self.frm_none,
            "Histogram Equalization": self.frm_none,
            "CLAHE": self.frm_clahe,
            "Gamma": self.frm_gamma,
            "Logarithm": self.frm_none,
            "Contrast Stretching": self.frm_cs,
            "Gray-Level Slicing": self.frm_slice,
        }
        mapping.get(op_name, self.frm_none).pack(fill=tk.X)

        # 3) Nếu chọn Enhance = CLAHE thì cũng hiện tham số CLAHE
        if hasattr(self, "enhance_var") and self.enhance_var.get() == "CLAHE":
            if self.frm_clahe not in self.param_stack.pack_slaves():
                self.frm_clahe.pack(fill=tk.X)

        # 4) Nếu bật chain các phép có tham số, hiện thêm slider tương ứng
        if hasattr(self, "ch_gam") and self.ch_gam.get():
            if self.frm_gamma not in self.param_stack.pack_slaves():
                self.frm_gamma.pack(fill=tk.X)
        if hasattr(self, "ch_cs") and self.ch_cs.get():
            if self.frm_cs not in self.param_stack.pack_slaves():
                self.frm_cs.pack(fill=tk.X)
        if hasattr(self, "ch_slice") and self.ch_slice.get():
            if self.frm_slice not in self.param_stack.pack_slaves():
                self.frm_slice.pack(fill=tk.X)

    # -------------- Processing (trên ảnh xám) --------------
    def update_processed(self):
        if self.img_bgr_orig is None: return
        op = self.op_var.get()
        gray = cv2.cvtColor(self.img_bgr_orig, cv2.COLOR_BGR2GRAY)

        # --- Enhance selection ---
        enh   = self.enhance_var.get() if hasattr(self, "enhance_var") else "None"
        order = self.combine_order.get() if hasattr(self, "combine_order") else "Enhance→Base"
        clip  = float(self.clahe_clip.get()) if hasattr(self, "clahe_clip") else 2.0
        tile  = int(self.clahe_tile.get()) if hasattr(self, "clahe_tile") else 8

        def apply_enhance(img_gray):
            if enh == "HE":
                return self.hist_equalization(img_gray)
            elif enh == "CLAHE":
                return self._clahe_gray(img_gray, clip, tile)
            return img_gray

        # 1) Enhance trước (tuỳ chọn)
        src = apply_enhance(gray) if (enh != "None" and order == "Enhance→Base") else gray

        # 2) Base op
        if op in ("None", "Grayscale"):
            out_gray = src
        elif op == "Negative":
            out_gray = self.negative(src)
        elif op == "Histogram Equalization":
            out_gray = self.hist_equalization(src)
        elif op == "CLAHE":
            out_gray = self._clahe_gray(src, clip, tile)
        elif op == "Gamma":
            out_gray = self._gamma_gray(src, float(self.gamma_var.get()))
        elif op == "Logarithm":
            out_gray = self._log_gray(src)
        elif op == "Contrast Stretching":
            out_gray = self._contrast_stretch(
                src,
                int(self.cs_r1.get()), int(self.cs_s1.get()),
                int(self.cs_r2.get()), int(self.cs_s2.get())
            )
        elif op == "Gray-Level Slicing":
            out_gray = self.slicedGreyScale(
                src, int(self.sl_t1.get()), int(self.sl_t2.get()), bool(self.keep_bg.get())
            )
        else:
            out_gray = src

        # 3) Enhance sau (tuỳ chọn)
        if enh != "None" and order == "Base→Enhance":
            out_gray = apply_enhance(out_gray)

        # 4) Chain extra ops (theo thứ tự cố định)
        neg_at_end = self.neg_after_var.get() if hasattr(self, "neg_after_var") else False

        if hasattr(self, "ch_log") and self.ch_log.get():
            out_gray = self._log_gray(out_gray)

        if hasattr(self, "ch_gam") and self.ch_gam.get():
            out_gray = self._gamma_gray(out_gray, float(self.gamma_var.get()))

        if hasattr(self, "ch_cs") and self.ch_cs.get():
            out_gray = self._contrast_stretch(
                out_gray,
                int(self.cs_r1.get()), int(self.cs_s1.get()),
                int(self.cs_r2.get()), int(self.cs_s2.get())
            )

        if hasattr(self, "ch_slice") and self.ch_slice.get():
            out_gray = self.slicedGreyScale(
                out_gray, int(self.sl_t1.get()), int(self.sl_t2.get()), bool(self.keep_bg.get())
            )

        if hasattr(self, "ch_neg") and self.ch_neg.get() and not neg_at_end:
            out_gray = self.negative(out_gray)

        if neg_at_end:
            out_gray = self.negative(out_gray)

        self.img_bgr_work = self._ensure_bgr(out_gray)
        self.update_previews()

    # ---- các hàm xử lý xám ----
    def negative(self, gray):
        return 255 - gray

    def hist_equalization(self, gray, L=256):
        if gray.dtype == np.uint8:
            img_u8 = gray
        else:
            img_u8 = np.clip(gray, 0, 255).astype(np.uint8)
        hist = np.bincount(img_u8.ravel(), minlength=L)
        cdf  = np.cumsum(hist) / hist.sum()
        T    = np.round((L - 1) * cdf).astype(np.uint8)
        img_eq = T[img_u8]
        return img_eq

    def _clahe_gray(self, gray, clip, tile):
        tile = max(2, int(tile))
        clahe = cv2.createCLAHE(clipLimit=max(1.0, float(clip)), tileGridSize=(tile, tile))
        return clahe.apply(gray)

    def _gamma_gray(self, gray, gamma):
        gamma = max(0.01, float(gamma))
        f = gray.astype(np.float32) / 255.0           # [0,1]
        g = np.power(f, gamma)                        # gamma<1 sáng hơn; >1 tối hơn
        out = np.clip(g, 0.0, 1.0) * 255.0
        return out.astype(np.uint8)

    def _log_gray(self, gray):
        r = gray.astype(np.float32)                   # np.log1p(x) = ln(1+x)
        c = 255.0 / np.log1p(max(float(r.max()), 1.0))
        out = c * np.log1p(r)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _contrast_stretch(self, gray, r1, s1, r2, s2):
        g = gray.astype(np.float32)
        r1, r2 = int(min(r1, r2)), int(max(r1, r2))
        s1, s2 = int(s1), int(s2)

        out = np.empty_like(g)
        m1 = g <= r1
        m2 = (g > r1) & (g <= r2)
        m3 = g > r2

        out[m1] = (s1 / max(r1, 1e-6)) * g[m1]
        out[m2] = ((s2 - s1) / max((r2 - r1), 1e-6)) * (g[m2] - r1) + s1
        out[m3] = ((255 - s2) / max((255 - r2), 1e-6)) * (g[m3] - r2) + s2
        return np.clip(out, 0, 255).astype(np.uint8)

    def slicedGreyScale(self, gray, T1=150, T2=240, keep_bg=True):
        t1, t2 = int(min(T1, T2)), int(max(T1, T2))
        mask = (gray > t1) & (gray < t2)
        out = gray.copy() if keep_bg else np.zeros_like(gray, np.uint8)
        out[mask] = np.clip(gray[mask].astype(np.int16) + 60, 0, 255).astype(np.uint8)
        return out

    def _ensure_bgr(self, arr):
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR) if arr.ndim == 2 else arr

    # -------------- Display --------------
    def update_previews(self):
        if self.img_bgr_orig is None:
            self._show_placeholder(); return
        self._draw_on_canvas(self.panel_orig, self.img_bgr_orig)
        self._draw_on_canvas(self.panel_proc, self.img_bgr_work if self.img_bgr_work is not None else self.img_bgr_orig)

    def _redraw(self):
        if self.img_bgr_orig is not None:
            self.update_previews()

    def _draw_on_canvas(self, canvas: tk.Canvas, bgr_img):
        cw = max(10, canvas.winfo_width())
        ch = max(10, canvas.winfo_height())
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        rgb_resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        pil = Image.fromarray(rgb_resized)
        tkimg = ImageTk.PhotoImage(pil)
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=tkimg, anchor="center")
        if canvas is self.panel_orig: self.tk_img_left = tkimg
        else: self.tk_img_right = tkimg

if __name__ == "__main__":
    app = ImageApp()
    app.mainloop()
