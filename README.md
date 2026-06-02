# Interactive Graph Cut Segmentation

Coursework project for Image Processing.

This project creates an interactive foreground/background segmentation tool. Users mark foreground and background seeds, then the program builds a graph and computes a segmentation using graph cut.

## Features

- Interactive foreground/background seed marking
- Gaussian Mixture Models for foreground and background color likelihoods
- Pixel graph construction with terminal links and neighborhood links
- Smoothness weighting based on color similarity
- Min-cut segmentation using NetworkX
- Result visualization with Matplotlib

## Tech Stack

- Python
- OpenCV
- NumPy
- scikit-learn
- NetworkX
- Matplotlib

## Run

```bash
pip install -r requirements.txt
python hw3_graph_cut_completed.py path/to/image.jpg
```

Left-click or drag to mark foreground seeds, right-click or drag to mark background seeds, then press any key in the image window to run segmentation.

