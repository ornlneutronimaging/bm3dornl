# BM3D GUI

GUI application for BM3D ring artifact removal in neutron imaging.

## Installation

```bash
pip install bm3dornl-gui
```

Or install together with the bm3dornl library:

```bash
pip install bm3dornl[gui]
```

## Usage

After installation, run:

```bash
bm3dornl-gui
```

## Features

- Load HDF5 and TIFF volume data
- Interactive slice viewing with adjustable window/level
- Multiple colormaps (Grayscale, Viridis, Plasma, etc.)
- BM3D processing with configurable parameters
- Side-by-side comparison of original and processed data
- Histogram analysis with ROI selection
- Export results to HDF5 or TIFF format

## Requirements

- Linux x86_64 or macOS ARM64
- No additional dependencies (HDF5 is bundled)

## License

MIT License - see the main [bm3dornl repository](https://github.com/ornlneutronimaging/bm3dornl).
