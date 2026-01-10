use hdf5_metno::File as H5File;
use ndarray::Array3;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use tiff::decoder::{Decoder, DecodingResult};
use tiff::ColorType;

use super::Volume3D;

#[derive(Debug)]
pub enum DataLoadError {
    IoError(String),
    Hdf5Error(String),
    TiffError(String),
    InvalidDimensions(String),
    UnsupportedDataType(String),
}

impl std::fmt::Display for DataLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError(s) => write!(f, "IO error: {}", s),
            Self::Hdf5Error(s) => write!(f, "HDF5 error: {}", s),
            Self::TiffError(s) => write!(f, "TIFF error: {}", s),
            Self::InvalidDimensions(s) => write!(f, "Invalid dimensions: {}", s),
            Self::UnsupportedDataType(s) => write!(f, "Unsupported data type: {}", s),
        }
    }
}

/// Load a 3D dataset from an HDF5 file.
pub fn load_hdf5_dataset(path: &Path, dataset_path: &str) -> Result<Volume3D, DataLoadError> {
    let file = H5File::open(path).map_err(|e| DataLoadError::Hdf5Error(e.to_string()))?;

    let dataset = file
        .dataset(dataset_path)
        .map_err(|e| DataLoadError::Hdf5Error(e.to_string()))?;

    let shape = dataset.shape();
    if shape.len() != 3 {
        return Err(DataLoadError::InvalidDimensions(format!(
            "Expected 3D dataset, got {}D with shape {:?}",
            shape.len(),
            shape
        )));
    }

    // Try to read as various types and convert to f32
    let data: Array3<f32> = if let Ok(arr) = dataset.read::<f32, ndarray::Ix3>() {
        arr
    } else if let Ok(arr) = dataset.read::<f64, ndarray::Ix3>() {
        arr.mapv(|v| v as f32)
    } else if let Ok(arr) = dataset.read::<u16, ndarray::Ix3>() {
        arr.mapv(|v| v as f32)
    } else if let Ok(arr) = dataset.read::<i16, ndarray::Ix3>() {
        arr.mapv(|v| v as f32)
    } else if let Ok(arr) = dataset.read::<u32, ndarray::Ix3>() {
        arr.mapv(|v| v as f32)
    } else if let Ok(arr) = dataset.read::<i32, ndarray::Ix3>() {
        arr.mapv(|v| v as f32)
    } else if let Ok(arr) = dataset.read::<u8, ndarray::Ix3>() {
        arr.mapv(|v| v as f32)
    } else {
        return Err(DataLoadError::UnsupportedDataType(
            "Could not read dataset as f32, f64, u16, i16, u32, i32, or u8".to_string(),
        ));
    };

    Ok(Volume3D::new(data))
}

/// Load a multi-page TIFF as a 3D volume.
/// Shape will be [num_pages, height, width].
pub fn load_tiff_stack(path: &Path) -> Result<Volume3D, DataLoadError> {
    let file = File::open(path).map_err(|e| DataLoadError::IoError(e.to_string()))?;
    let reader = BufReader::new(file);
    let mut decoder = Decoder::new(reader).map_err(|e| DataLoadError::TiffError(e.to_string()))?;

    let mut pages: Vec<Vec<f32>> = Vec::new();
    let mut width = 0usize;
    let mut height = 0usize;

    loop {
        let (w, h) = decoder
            .dimensions()
            .map_err(|e| DataLoadError::TiffError(e.to_string()))?;
        let color_type = decoder
            .colortype()
            .map_err(|e| DataLoadError::TiffError(e.to_string()))?;

        // Only support grayscale for now
        if !matches!(
            color_type,
            ColorType::Gray(8) | ColorType::Gray(16) | ColorType::Gray(32)
        ) {
            return Err(DataLoadError::UnsupportedDataType(format!(
                "Unsupported TIFF color type: {:?}. Only grayscale supported.",
                color_type
            )));
        }

        if pages.is_empty() {
            width = w as usize;
            height = h as usize;
        } else if w as usize != width || h as usize != height {
            return Err(DataLoadError::InvalidDimensions(format!(
                "TIFF pages have inconsistent dimensions: expected {}x{}, got {}x{}",
                width, height, w, h
            )));
        }

        let image_data = decoder
            .read_image()
            .map_err(|e| DataLoadError::TiffError(e.to_string()))?;

        let page_f32: Vec<f32> = match image_data {
            DecodingResult::U8(data) => data.into_iter().map(|v| v as f32).collect(),
            DecodingResult::U16(data) => data.into_iter().map(|v| v as f32).collect(),
            DecodingResult::U32(data) => data.into_iter().map(|v| v as f32).collect(),
            DecodingResult::U64(data) => data.into_iter().map(|v| v as f32).collect(),
            DecodingResult::I8(data) => data.into_iter().map(|v| v as f32).collect(),
            DecodingResult::I16(data) => data.into_iter().map(|v| v as f32).collect(),
            DecodingResult::I32(data) => data.into_iter().map(|v| v as f32).collect(),
            DecodingResult::I64(data) => data.into_iter().map(|v| v as f32).collect(),
            DecodingResult::F32(data) => data,
            DecodingResult::F64(data) => data.into_iter().map(|v| v as f32).collect(),
        };

        pages.push(page_f32);

        // Try to move to next page
        if decoder.more_images() {
            if decoder.next_image().is_err() {
                break;
            }
        } else {
            break;
        }
    }

    if pages.is_empty() {
        return Err(DataLoadError::TiffError(
            "No pages found in TIFF".to_string(),
        ));
    }

    let num_pages = pages.len();

    // Create 3D array [num_pages, height, width]
    let mut data = Array3::<f32>::zeros((num_pages, height, width));
    for (page_idx, page_data) in pages.into_iter().enumerate() {
        for (pixel_idx, val) in page_data.into_iter().enumerate() {
            let y = pixel_idx / width;
            let x = pixel_idx % width;
            if y < height && x < width {
                data[[page_idx, y, x]] = val;
            }
        }
    }

    Ok(Volume3D::new(data))
}

/// Information about an HDF5 entry for tree display.
#[derive(Debug, Clone)]
pub enum Hdf5Entry {
    Group {
        name: String,
        path: String,
        children: Vec<Hdf5Entry>,
    },
    Dataset {
        name: String,
        path: String,
        shape: Vec<usize>,
        dtype: String,
    },
}

impl Hdf5Entry {
    pub fn name(&self) -> &str {
        match self {
            Self::Group { name, .. } => name,
            Self::Dataset { name, .. } => name,
        }
    }

    pub fn path(&self) -> &str {
        match self {
            Self::Group { path, .. } => path,
            Self::Dataset { path, .. } => path,
        }
    }
}

/// Build a tree structure of an HDF5 file.
pub fn build_hdf5_tree(path: &Path) -> Result<Vec<Hdf5Entry>, DataLoadError> {
    let file = H5File::open(path).map_err(|e| DataLoadError::Hdf5Error(e.to_string()))?;
    build_group_tree(&file, "/")
}

fn build_group_tree(
    group: &hdf5_metno::Group,
    prefix: &str,
) -> Result<Vec<Hdf5Entry>, DataLoadError> {
    let member_names = group
        .member_names()
        .map_err(|e| DataLoadError::Hdf5Error(e.to_string()))?;

    let mut entries = Vec::new();

    for name in member_names {
        let full_path = if prefix == "/" {
            format!("/{}", name)
        } else {
            format!("{}/{}", prefix, name)
        };

        if let Ok(subgroup) = group.group(&name) {
            let children = build_group_tree(&subgroup, &full_path)?;
            entries.push(Hdf5Entry::Group {
                name: name.clone(),
                path: full_path,
                children,
            });
        } else if let Ok(dataset) = group.dataset(&name) {
            let shape: Vec<usize> = dataset.shape().to_vec();
            let dtype = dataset
                .dtype()
                .map(|d| format!("{:?}", d))
                .unwrap_or_else(|_| "unknown".to_string());
            entries.push(Hdf5Entry::Dataset {
                name: name.clone(),
                path: full_path,
                shape,
                dtype,
            });
        }
    }

    Ok(entries)
}
