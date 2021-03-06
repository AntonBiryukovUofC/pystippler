# Pystippler
My attempt to set up a stippling procedure in Python. Inspired (and built on top of) by https://github.com/ReScience-Archives/Rougier-2017

Built using `skimage`, `scikit-learn` and `streamlit`.

![lumi](lumi.gif)

# How to run
`streamlit run streamlit_stipple.py` will launch an interactive Streamlit webinterface, where you can explore stippling
with various parameters.

# Dependencies

- `streamlit` for running the webapp
- `altair` for visualization
- `imageio`, `scikit-image` for image work
- `scikit-learn` for color space quantization
- `tqdm` for progress bars
- `numba` for some speedups
- others are same as in the `Rougier-2017`
