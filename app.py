# app.py  (FINAL)
# WB Quant（最终稳定版｜单ROI拖动记录 + 置换检验p值）
# 依赖建议：
#   pip install "streamlit==1.31.1" "streamlit-drawable-canvas==0.9.3" "pillow" "numpy<2" "pandas" "matplotlib" "scipy"
#
# 运行：
#   streamlit run app.py

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from PIL import Image, ImageFilter
from streamlit_drawable_canvas import st_canvas

try:
    from scipy.stats import mannwhitneyu
except Exception:
    mannwhitneyu = None

st.set_page_config(page_title="WB Quant (Stable)", page_icon="🧪", layout="wide")


# -------------------------
# Utils: image IO & preprocess
# -------------------------
def _pil_to_gray_np(pil_img: Image.Image) -> np.ndarray:
    g = pil_img.convert("L")
    return np.array(g, dtype=np.float32)


def preprocess_gray(arr: np.ndarray, invert: bool = True, blur_radius: float = 1.2) -> np.ndarray:
    x = arr.astype(np.float32)
    if invert:
        x = float(x.max()) - x

    if blur_radius and blur_radius > 0:
        pil = Image.fromarray(np.clip(x, 0, 255).astype(np.uint8))
        pil = pil.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
        x = np.array(pil, dtype=np.float32)
    return x


def to_uint8_for_display(arr: np.ndarray) -> np.ndarray:
    x = arr.astype(np.float32)
    x = x - float(np.min(x))
    mx = float(np.max(x))
    if mx > 0:
        x = x / mx
    return (x * 255.0).astype(np.uint8)


@st.cache_data(show_spinner=False)
def load_image_bytes(file_bytes: bytes, invert: bool = True, blur_radius: float = 1.2):
    pil = Image.open(io.BytesIO(file_bytes))
    gray = _pil_to_gray_np(pil)
    gray = preprocess_gray(gray, invert=invert, blur_radius=blur_radius)
    u8 = to_uint8_for_display(gray)

    # ✅ 关键：canvas 背景图必须与 width/height 完全一致（原尺寸），RGB 最稳
    pil_bg = Image.fromarray(u8).convert("RGB")
    return u8.astype(np.float32), pil_bg


# -------------------------
# ROI + quant
# -------------------------
def clamp_rect_to_image(rect, W, H):
    left = float(rect.get("left", 0.0))
    top = float(rect.get("top", 0.0))
    width = float(rect.get("width", 0.0)) * float(rect.get("scaleX", 1.0))
    height = float(rect.get("height", 0.0)) * float(rect.get("scaleY", 1.0))

    left = max(0.0, min(left, W - 2))
    top = max(0.0, min(top, H - 2))
    width = max(2.0, min(width, W - left))
    height = max(2.0, min(height, H - top))
    return left, top, width, height


def rect_intden_with_ring_bg(img: np.ndarray, rect, bg_pad: int = 6) -> float:
    H, W = img.shape
    left, top, width, height = clamp_rect_to_image(rect, W, H)

    x1 = int(round(left))
    y1 = int(round(top))
    x2 = int(round(left + width))
    y2 = int(round(top + height))

    x1 = max(0, min(x1, W - 2))
    y1 = max(0, min(y1, H - 2))
    x2 = max(x1 + 1, min(x2, W))
    y2 = max(y1 + 1, min(y2, H))

    inner = img[y1:y2, x1:x2]
    inner_mean = float(inner.mean())
    inner_area = int(inner.size)

    pad = int(max(0, bg_pad))
    ox1 = max(0, x1 - pad)
    oy1 = max(0, y1 - pad)
    ox2 = min(W, x2 + pad)
    oy2 = min(H, y2 + pad)

    outer = img[oy1:oy2, ox1:ox2]
    mask = np.ones(outer.shape, dtype=bool)

    ix1 = x1 - ox1
    iy1 = y1 - oy1
    ix2 = ix1 + (x2 - x1)
    iy2 = iy1 + (y2 - y1)
    mask[iy1:iy2, ix1:ix2] = False

    bg_pixels = outer[mask]
    bg_val = float(np.median(bg_pixels)) if bg_pixels.size else 0.0

    return (inner_mean - bg_val) * inner_area


def extract_last_rect(canvas_json):
    if not canvas_json:
        return None
    objs = canvas_json.get("objects", [])
    rects = [o for o in objs if o.get("type") in ("rect", "Rect", "rectangle")]
    return rects[-1] if rects else None


def lock_rect_size(rect_obj, template_wh):
    if rect_obj is None or template_wh is None:
        return rect_obj
    w, h = template_wh
    r = dict(rect_obj)
    r["width"] = float(w)
    r["height"] = float(h)
    r["scaleX"] = 1.0
    r["scaleY"] = 1.0
    return r


def shift_rect(rect_obj, dx=0.0, dy=0.0, W=None, H=None):
    if rect_obj is None:
        return None
    r = dict(rect_obj)
    r["left"] = float(r.get("left", 0.0)) + float(dx)
    r["top"] = float(r.get("top", 0.0)) + float(dy)
    if W is not None and H is not None:
        left, top, width, height = clamp_rect_to_image(r, W, H)
        r["left"], r["top"] = left, top
        r["width"], r["height"] = float(width), float(height)
        r["scaleX"], r["scaleY"] = 1.0, 1.0
    return r


def single_rect_drawing(rect_obj):
    if rect_obj is None:
        return {"version": "5.3.0", "objects": []}
    r = dict(rect_obj)
    r["type"] = "rect"
    r["fill"] = "rgba(0,0,0,0)"
    r["strokeWidth"] = 2
    r["selectable"] = True
    r["evented"] = True
    return {"version": "5.3.0", "objects": [r]}


# -------------------------
# Stats
# -------------------------
def p_to_star(p: float) -> str:
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def permutation_pvalue(a: np.ndarray, b: np.ndarray, n_perm: int = 20000, seed: int = 0, stat: str = "mean"):
    """
    Two-sided permutation test p-value.
    stat: "mean" or "median"
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 1 or len(b) < 1:
        return np.nan

    rng = np.random.default_rng(seed)
    x = np.concatenate([a, b])
    n_a = len(a)

    if stat == "median":
        obs = abs(np.median(a) - np.median(b))
    else:
        obs = abs(np.mean(a) - np.mean(b))

    cnt = 0
    for _ in range(int(n_perm)):
        idx = rng.permutation(len(x))
        aa = x[idx[:n_a]]
        bb = x[idx[n_a:]]
        if stat == "median":
            val = abs(np.median(aa) - np.median(bb))
        else:
            val = abs(np.mean(aa) - np.mean(bb))
        if val >= obs:
            cnt += 1

    # +1 smoothing
    p = (cnt + 1) / (n_perm + 1)
    return p


# -------------------------
# Plot (PPT-friendly) + p-value
# -------------------------
def make_ppt_plot(dff: pd.DataFrame, value_col: str, group_col: str, title: str,
                  test_method: str = "Permutation", n_perm: int = 20000, perm_stat: str = "mean"):
    groups = sorted([g for g in dff[group_col].dropna().unique().tolist()])
    xs = np.arange(len(groups))

    fig, ax = plt.subplots(figsize=(4.3, 3.0), dpi=220)

    means, sems, all_vals = [], [], []
    for g in groups:
        vals = dff.loc[dff[group_col] == g, value_col].astype(float).values
        vals = vals[np.isfinite(vals)]
        all_vals.append(vals)
        means.append(vals.mean() if len(vals) else np.nan)
        sems.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)

    ax.bar(xs, means, yerr=sems, capsize=3, alpha=0.92, linewidth=0.8)

    for i, vals in enumerate(all_vals):
        if len(vals) == 0:
            continue
        jitter = (np.random.rand(len(vals)) - 0.5) * 0.18
        ax.scatter(np.full_like(vals, xs[i], dtype=float) + jitter, vals, s=14, zorder=3)

    ax.set_xticks(xs)
    ax.set_xticklabels(groups, fontsize=8)
    ax.set_ylabel(value_col, fontsize=9)
    ax.set_title(title, fontsize=10, pad=10)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)

    # p-value label
    if len(groups) == 2:
        a, b = all_vals[0], all_vals[1]
        p = np.nan
        label = ""

        if test_method == "Mann-Whitney" and mannwhitneyu is not None:
            # exact/auto is discrete for tiny n; still provide as option
            p = mannwhitneyu(a, b, alternative="two-sided", method="auto").pvalue
            label = f"MWU p={p:.3g}"
        else:
            p = permutation_pvalue(a, b, n_perm=n_perm, seed=0, stat=perm_stat)
            label = f"Perm p={p:.3g}"

        if np.isfinite(p):
            y_max = float(np.nanmax(dff[value_col].astype(float).values))
            y = y_max * 1.22 if y_max > 0 else 1.0
            ax.plot([0, 0, 1, 1], [y * 0.98, y, y, y * 0.98], lw=1.0)
            ax.text(0.5, y * 1.01, f"{p_to_star(p)}  ({label})",
                    ha="center", va="bottom", fontsize=8)
            ax.set_ylim(top=y * 1.10)

    fig.tight_layout()
    return fig


# -------------------------
# Session state
# -------------------------
ss = st.session_state
defaults = {
    "ref_vals": [],
    "tar_vals": [],
    "ref_rect": None,
    "tar_rect": None,
    "ref_template_wh": None,
    "tar_template_wh": None,
    "ref_lane_idx": 1,
    "tar_lane_idx": 1,
}
for k, v in defaults.items():
    if k not in ss:
        ss[k] = v


# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("基本信息")
    ref_name = st.text_input("内参蛋白名（Ref）", value="GAPDH")
    tar_name = st.text_input("目的蛋白名（Target）", value="Target")

    st.divider()
    st.header("参数")
    invert = st.checkbox("反相（让条带更亮）", value=True)
    blur_radius = st.slider("轻度平滑（radius）", min_value=0.0, max_value=3.0, value=1.2, step=0.1)

    n_lanes = st.number_input("lane 数（2-15）", min_value=2, max_value=15, value=4, step=1)
    bg_pad = st.number_input("背景环带宽度（px）", min_value=0, max_value=80, value=6, step=1)

    st.divider()
    st.header("丝滑记录")
    auto_shift = st.checkbox("记录后自动右移到下一个 lane", value=True)
    shift_dx = st.number_input("右移 dx（像素）", min_value=0, max_value=5000, value=120, step=5)
    shift_dy = st.number_input("下移 dy（可选）", min_value=-2000, max_value=2000, value=0, step=1)

    st.divider()
    st.header("分组/作图")
    group_text = st.text_input("每个 lane 的组名（逗号分隔）", value="CTRL,HFD,CTRL,HFD")
    metric = st.selectbox("作图指标", ["Target/Ref", "Target", "Ref"], index=0)

    st.divider()
    st.header("P值检验（两组）")
    test_method = st.selectbox("方法", ["Permutation", "Mann-Whitney"], index=0)
    n_perm = st.number_input("置换次数（Permutation）", min_value=2000, max_value=100000, value=20000, step=2000)
    perm_stat = st.selectbox("置换统计量", ["mean", "median"], index=0)
    st.caption("提示：样本量很小（如2vs2）时，Mann-Whitney p值会很离散，常见0.33；Permutation更平滑。")


# -------------------------
# Main
# -------------------------
st.title("WB Quant（最终稳定版：单ROI拖动→记录 + 置换检验）")
st.caption("第一次画框确定大小；之后只拖动同一个框到下一个条带 → 记录 →（可选）自动右移。")

u1, u2 = st.columns(2)
with u1:
    ref_file = st.file_uploader("上传内参图（Ref）", type=["tif", "tiff", "png", "jpg", "jpeg"])
with u2:
    tar_file = st.file_uploader("上传目标图（Target）", type=["tif", "tiff", "png", "jpg", "jpeg"])

if (ref_file is None) or (tar_file is None):
    st.info("请先上传内参图和目标图。")
    st.stop()

ref_img, ref_pil = load_image_bytes(ref_file.getvalue(), invert=invert, blur_radius=blur_radius)
tar_img, tar_pil = load_image_bytes(tar_file.getvalue(), invert=invert, blur_radius=blur_radius)

W_ref, H_ref = ref_pil.size
W_tar, H_tar = tar_pil.size

with st.expander("（可选）预览", expanded=False):
    st.write(f"Ref size: {W_ref}×{H_ref}")
    st.image(ref_pil, caption="Ref 预览", width=min(900, W_ref))
    st.write(f"Target size: {W_tar}×{H_tar}")
    st.image(tar_pil, caption="Target 预览", width=min(900, W_tar))

cA, cB, cC = st.columns([1.1, 1.4, 1.2])
with cA:
    if st.button("重置全部（结果+ROI）"):
        ss.ref_vals = []
        ss.tar_vals = []
        ss.ref_rect = None
        ss.tar_rect = None
        ss.ref_template_wh = None
        ss.tar_template_wh = None
        ss.ref_lane_idx = 1
        ss.tar_lane_idx = 1
        st.rerun()
with cB:
    if st.button("撤销一步（Ref+Target 各删最后一次记录）"):
        if ss.ref_vals:
            ss.ref_vals.pop()
            ss.ref_lane_idx = max(1, ss.ref_lane_idx - 1)
        if ss.tar_vals:
            ss.tar_vals.pop()
            ss.tar_lane_idx = max(1, ss.tar_lane_idx - 1)
        st.rerun()
with cC:
    st.write(f"Ref lane：{ss.ref_lane_idx}/{int(n_lanes)}；Target lane：{ss.tar_lane_idx}/{int(n_lanes)}")

st.divider()
st.subheader("框选与记录（先画框→再拖动）")

ref_mode = "rect" if ss.ref_rect is None else "transform"
tar_mode = "rect" if ss.tar_rect is None else "transform"

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"### Ref: {ref_name}")

    ref_canvas = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=2,
        stroke_color="#00E5FF",
        background_image=ref_pil,
        update_streamlit=True,
        drawing_mode=ref_mode,
        initial_drawing=single_rect_drawing(ss.ref_rect),
        height=H_ref,
        width=W_ref,
        key="ref_canvas_final",
    )

    ref_new = extract_last_rect(ref_canvas.json_data)
    if ref_new is not None:
        if ss.ref_template_wh is None:
            w = float(ref_new.get("width", 10.0)) * float(ref_new.get("scaleX", 1.0))
            h = float(ref_new.get("height", 10.0)) * float(ref_new.get("scaleY", 1.0))
            ss.ref_template_wh = (max(2.0, w), max(2.0, h))
        ss.ref_rect = lock_rect_size(ref_new, ss.ref_template_wh)

    r1, r2 = st.columns(2)
    with r1:
        if st.button("记录 Ref 当前 lane"):
            if ss.ref_rect is None:
                st.warning("Ref：请先画一个框。")
            elif ss.ref_lane_idx > int(n_lanes):
                st.warning("Ref：已记录满 lane 数。")
            else:
                val = rect_intden_with_ring_bg(ref_img, ss.ref_rect, bg_pad=int(bg_pad))
                ss.ref_vals.append(val)
                ss.ref_lane_idx += 1
                if auto_shift:
                    ss.ref_rect = shift_rect(ss.ref_rect, dx=float(shift_dx), dy=float(shift_dy), W=W_ref, H=H_ref)
                st.rerun()
    with r2:
        if st.button("Ref：清除ROI+结果"):
            ss.ref_rect = None
            ss.ref_template_wh = None
            ss.ref_vals = []
            ss.ref_lane_idx = 1
            st.rerun()

with col2:
    st.markdown(f"### Target: {tar_name}")

    tar_canvas = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=2,
        stroke_color="#00E5FF",
        background_image=tar_pil,
        update_streamlit=True,
        drawing_mode=tar_mode,
        initial_drawing=single_rect_drawing(ss.tar_rect),
        height=H_tar,
        width=W_tar,
        key="tar_canvas_final",
    )

    tar_new = extract_last_rect(tar_canvas.json_data)
    if tar_new is not None:
        if ss.tar_template_wh is None:
            w = float(tar_new.get("width", 10.0)) * float(tar_new.get("scaleX", 1.0))
            h = float(tar_new.get("height", 10.0)) * float(tar_new.get("scaleY", 1.0))
            ss.tar_template_wh = (max(2.0, w), max(2.0, h))
        ss.tar_rect = lock_rect_size(tar_new, ss.tar_template_wh)

    t1, t2 = st.columns(2)
    with t1:
        if st.button("记录 Target 当前 lane"):
            if ss.tar_rect is None:
                st.warning("Target：请先画一个框。")
            elif ss.tar_lane_idx > int(n_lanes):
                st.warning("Target：已记录满 lane 数。")
            else:
                val = rect_intden_with_ring_bg(tar_img, ss.tar_rect, bg_pad=int(bg_pad))
                ss.tar_vals.append(val)
                ss.tar_lane_idx += 1
                if auto_shift:
                    ss.tar_rect = shift_rect(ss.tar_rect, dx=float(shift_dx), dy=float(shift_dy), W=W_tar, H=H_tar)
                st.rerun()
    with t2:
        if st.button("Target：清除ROI+结果"):
            ss.tar_rect = None
            ss.tar_template_wh = None
            ss.tar_vals = []
            ss.tar_lane_idx = 1
            st.rerun()


# -------------------------
# Results table
# -------------------------
st.divider()
st.subheader("结果表 / 导出")

n = int(n_lanes)
ref_vals = ss.ref_vals[:n] + [np.nan] * max(0, n - len(ss.ref_vals))
tar_vals = ss.tar_vals[:n] + [np.nan] * max(0, n - len(ss.tar_vals))

df = pd.DataFrame({
    "Lane": np.arange(1, n + 1),
    f"{ref_name} (Ref)": ref_vals,
    f"{tar_name} (Target)": tar_vals,
})
df["Target/Ref"] = df[f"{tar_name} (Target)"] / (df[f"{ref_name} (Ref)"] + 1e-9)

groups = [g.strip() for g in group_text.split(",")] if group_text.strip() else []
df["Group"] = groups if len(groups) == n else np.nan

st.dataframe(df)

st.download_button(
    "下载结果 CSV",
    data=df.to_csv(index=False).encode("utf-8-sig"),
    file_name="wb_quant_results.csv",
    mime="text/csv",
)


# -------------------------
# Plot + Download (fixed blank export)
# -------------------------
st.divider()
st.subheader("统计图（小尺寸｜适合PPT）")

if df["Group"].notna().any():
    if metric == "Target":
        value_col = f"{tar_name} (Target)"
        title = f"{tar_name} by Group"
    elif metric == "Ref":
        value_col = f"{ref_name} (Ref)"
        title = f"{ref_name} by Group"
    else:
        value_col = "Target/Ref"
        title = f"{tar_name} normalized to {ref_name}"

    dff = df.dropna(subset=[value_col, "Group"])
    if len(dff) == 0:
        st.info("当前分组下没有可用数据（可能还没记录完或有 NaN）。")
    else:
        fig = make_ppt_plot(
            dff,
            value_col=value_col,
            group_col="Group",
            title=title,
            test_method=test_method,
            n_perm=int(n_perm),
            perm_stat=perm_stat,
        )

        # ✅ 先保存，再显示（避免下载空白）
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        png_bytes = buf.getvalue()

        st.pyplot(fig, clear_figure=False)
        plt.close(fig)

        st.download_button(
            "下载统计图 PNG（PPT用）",
            data=png_bytes,
            file_name="wb_plot.png",
            mime="image/png",
        )
else:
    st.info("要画图：左侧填每个 lane 的组名（逗号分隔），数量必须等于 lane 数。")
