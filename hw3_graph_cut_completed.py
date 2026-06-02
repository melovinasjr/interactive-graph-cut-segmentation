# HW3 Graph Cut - Completed Version
# Requirements used: numpy, opencv-python, scikit-learn, networkx, matplotlib
# Install on Windows if needed:
# pip install numpy opencv-python scikit-learn networkx matplotlib

import argparse
import os
import cv2
import numpy as np
import networkx as nx
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


def resize_keep_aspect(img_rgb, max_size=180):
    """Resize image to keep graph size manageable for NetworkX."""
    h, w = img_rgb.shape[:2]
    scale = min(1.0, float(max_size) / max(h, w))
    if scale < 1.0:
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img_small = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        img_small = img_rgb.copy()
    return img_small, scale


def draw_mask_overlay(img_rgb, mask, radius=2):
    """Create marked image: foreground seeds are green, background seeds are red."""
    marked = img_rgb.copy()
    fg = mask == 1
    bg = mask == 0
    marked[fg] = (0, 255, 0)
    marked[bg] = (255, 0, 0)
    return marked


def interactive_marking(img_rgb, radius=3):
    """Let the user mark foreground/background using mouse clicks.

    Left mouse button  = foreground seed
    Right mouse button = background seed
    Press any key when finished.
    """
    drawing_mask = np.full(img_rgb.shape[:2], 2, dtype=np.uint8)  # 2 means unknown
    disp_img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    window_name = "Mark: left=foreground, right=background, press any key when done"

    def mouse_callback(event, x, y, flags, param):
        nonlocal disp_img_bgr, drawing_mask
        if event == cv2.EVENT_LBUTTONDOWN or (flags & cv2.EVENT_FLAG_LBUTTON):
            cv2.circle(disp_img_bgr, (x, y), radius, (0, 255, 0), -1)
            cv2.circle(drawing_mask, (x, y), radius, 1, -1)
        elif event == cv2.EVENT_RBUTTONDOWN or (flags & cv2.EVENT_FLAG_RBUTTON):
            cv2.circle(disp_img_bgr, (x, y), radius, (0, 0, 255), -1)
            cv2.circle(drawing_mask, (x, y), radius, 0, -1)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("Use LEFT click/drag for foreground and RIGHT click/drag for background.")
    print("Press any key inside the image window when finished.")

    while True:
        cv2.imshow(window_name, disp_img_bgr)
        if cv2.waitKey(20) != -1:
            break
    cv2.destroyAllWindows()
    return drawing_mask, cv2.cvtColor(disp_img_bgr, cv2.COLOR_BGR2RGB)


def fit_gmm(seed_colors, max_components=3):
    """Fit a Gaussian Mixture Model to seed RGB colors."""
    n_samples = len(seed_colors)
    n_components = max(1, min(max_components, n_samples))
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        reg_covar=1e-5,
        random_state=0,
    )
    gmm.fit(seed_colors)
    return gmm


def normalize_costs(fg_cost, bg_cost, data_weight=8.0):
    """Robustly normalize data costs to keep graph capacities stable."""
    stacked = np.concatenate([fg_cost.ravel(), bg_cost.ravel()])
    lo, hi = np.percentile(stacked, [1, 99])
    if hi <= lo:
        hi = lo + 1e-6
    fg = np.clip((fg_cost - lo) / (hi - lo), 0.0, 1.0) * data_weight
    bg = np.clip((bg_cost - lo) / (hi - lo), 0.0, 1.0) * data_weight
    return fg, bg


def compute_boundary_parameters(img_float):
    """Compute beta for Gaussian n-link similarity weight."""
    diffs = []
    diffs.append(img_float[:, 1:, :] - img_float[:, :-1, :])
    diffs.append(img_float[1:, :, :] - img_float[:-1, :, :])
    diff_sq = np.concatenate([d.reshape(-1, 3) for d in diffs], axis=0)
    mean_sq = np.mean(np.sum(diff_sq * diff_sq, axis=1))
    beta = 1.0 / (2.0 * mean_sq + 1e-12)
    return beta


def build_graph(img_rgb, seed_mask, data_weight=8.0, smooth_weight=35.0, hard_weight=1e6):
    """Build the directed graph for Graph Cut.

    Nodes:
        each pixel is one node using index y*w + x
        's' is the foreground/source terminal
        't' is the background/sink terminal

    Edges:
        t-links connect terminal nodes and pixel nodes using GMM data costs
        n-links connect neighboring pixels using Gaussian color similarity
    """
    h, w = img_rgb.shape[:2]
    img_float = img_rgb.astype(np.float64) / 255.0
    pixels = img_float.reshape(-1, 3)

    fg_pixels = img_float[seed_mask == 1].reshape(-1, 3)
    bg_pixels = img_float[seed_mask == 0].reshape(-1, 3)

    if len(fg_pixels) < 2 or len(bg_pixels) < 2:
        raise ValueError("Please mark at least a few foreground and background pixels.")

    fg_gmm = fit_gmm(fg_pixels)
    bg_gmm = fit_gmm(bg_pixels)

    # Region property/data term from the GMM negative log-likelihood.
    fg_cost = -fg_gmm.score_samples(pixels).reshape(h, w)
    bg_cost = -bg_gmm.score_samples(pixels).reshape(h, w)
    fg_cost, bg_cost = normalize_costs(fg_cost, bg_cost, data_weight=data_weight)

    # Force user-labeled seeds to stay with their chosen label.
    # Source side means foreground; sink side means background.
    fg_cost[seed_mask == 1] = 0.0
    bg_cost[seed_mask == 1] = hard_weight
    fg_cost[seed_mask == 0] = hard_weight
    bg_cost[seed_mask == 0] = 0.0

    G = nx.DiGraph()
    G.add_node('s')
    G.add_node('t')

    # t-links:
    # If a pixel becomes background, edge s->p is cut, so capacity is background cost.
    # If a pixel becomes foreground, edge p->t is cut, so capacity is foreground cost.
    for y in range(h):
        for x in range(w):
            idx = y * w + x
            G.add_edge('s', idx, capacity=float(bg_cost[y, x]))
            G.add_edge(idx, 't', capacity=float(fg_cost[y, x]))

    # n-links: high capacity means neighboring pixels prefer to share the same label.
    beta = compute_boundary_parameters(img_float)
    for y in range(h):
        for x in range(w):
            idx = y * w + x
            current = img_float[y, x]
            if x + 1 < w:
                idx2 = y * w + (x + 1)
                diff_sq = np.sum((current - img_float[y, x + 1]) ** 2)
                weight = smooth_weight * np.exp(-beta * diff_sq)
                G.add_edge(idx, idx2, capacity=float(weight))
                G.add_edge(idx2, idx, capacity=float(weight))
            if y + 1 < h:
                idx2 = (y + 1) * w + x
                diff_sq = np.sum((current - img_float[y + 1, x]) ** 2)
                weight = smooth_weight * np.exp(-beta * diff_sq)
                G.add_edge(idx, idx2, capacity=float(weight))
                G.add_edge(idx2, idx, capacity=float(weight))

    return G


def graph_cut_segment(img_rgb, seed_mask, data_weight=8.0, smooth_weight=35.0):
    """Run Max-Flow/Min-Cut and return a binary foreground mask."""
    h, w = img_rgb.shape[:2]
    print("Building graph...")
    G = build_graph(img_rgb, seed_mask, data_weight=data_weight, smooth_weight=smooth_weight)
    print("Running NetworkX minimum_cut...")
    cut_value, partition = nx.minimum_cut(G, 's', 't')
    reachable, non_reachable = partition

    mask = np.zeros(h * w, dtype=np.uint8)
    for node in reachable:
        if isinstance(node, int):
            mask[node] = 1
    mask = mask.reshape(h, w)
    print(f"Done. Cut value = {cut_value:.4f}")
    return mask


def save_results(output_dir, base_name, original_rgb, marked_rgb, mask_small, scale):
    """Save original, marked, and segmented result images."""
    os.makedirs(output_dir, exist_ok=True)

    h0, w0 = original_rgb.shape[:2]
    if scale < 1.0:
        mask_full = cv2.resize(mask_small, (w0, h0), interpolation=cv2.INTER_NEAREST)
    else:
        mask_full = mask_small

    result_rgb = original_rgb * mask_full[:, :, None]

    original_path = os.path.join(output_dir, f"{base_name}_01_original.png")
    marked_path = os.path.join(output_dir, f"{base_name}_02_marked.png")
    result_path = os.path.join(output_dir, f"{base_name}_03_result.png")

    cv2.imwrite(original_path, cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(marked_path, cv2.cvtColor(marked_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(result_path, cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR))

    return original_path, marked_path, result_path


def main():
    parser = argparse.ArgumentParser(description="Interactive Graph Cut segmentation using GMM and NetworkX min-cut.")
    parser.add_argument("image_path", help="Path to the input image.")
    parser.add_argument("--output_dir", default="graphcut_outputs", help="Folder where output images will be saved.")
    parser.add_argument("--max_size", type=int, default=180, help="Maximum side length used for graph construction.")
    parser.add_argument("--radius", type=int, default=3, help="Brush radius for foreground/background marking.")
    parser.add_argument("--data_weight", type=float, default=8.0, help="Weight of GMM region/data term.")
    parser.add_argument("--smooth_weight", type=float, default=35.0, help="Weight of boundary smoothness term.")
    args = parser.parse_args()

    img_bgr = cv2.imread(args.image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {args.image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_small, scale = resize_keep_aspect(img_rgb, max_size=args.max_size)
    seed_mask, marked_small = interactive_marking(img_small, radius=args.radius)

    mask_small = graph_cut_segment(
        img_small,
        seed_mask,
        data_weight=args.data_weight,
        smooth_weight=args.smooth_weight,
    )

    if scale < 1.0:
        marked_full = cv2.resize(marked_small, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        marked_full = marked_small

    base_name = os.path.splitext(os.path.basename(args.image_path))[0]
    original_path, marked_path, result_path = save_results(
        args.output_dir,
        base_name,
        img_rgb,
        marked_full,
        mask_small,
        scale,
    )

    result_rgb = cv2.cvtColor(cv2.imread(result_path), cv2.COLOR_BGR2RGB)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img_rgb)
    axs[0].set_title("Original Image")
    axs[0].axis("off")
    axs[1].imshow(marked_full)
    axs[1].set_title("Foreground/Background Marks")
    axs[1].axis("off")
    axs[2].imshow(result_rgb)
    axs[2].set_title("GraphCut Result")
    axs[2].axis("off")
    plt.tight_layout()
    plt.show()

    print("Saved files:")
    print(original_path)
    print(marked_path)
    print(result_path)


if __name__ == "__main__":
    main()
