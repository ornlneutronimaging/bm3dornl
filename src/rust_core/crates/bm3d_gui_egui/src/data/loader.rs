use hdf5_metno::File as H5File;
use ndarray::Array3;
use std::fs::{self, File};
use std::io::BufReader;
use std::path::Path;
use tiff::decoder::{Decoder, DecodingResult};
use tiff::ColorType;

use super::Volume3D;

/// Extract numeric chunks from a string for natural sorting.
/// Returns a vector of (is_numeric, value) pairs where value is either
/// the numeric value or the lowercase string chunk.
fn natural_sort_key(s: &str) -> Vec<(bool, i64, String)> {
    let mut result = Vec::new();
    let mut current_chunk = String::new();
    let mut is_numeric = false;

    for c in s.chars() {
        let c_is_digit = c.is_ascii_digit();
        if current_chunk.is_empty() {
            is_numeric = c_is_digit;
            current_chunk.push(c);
        } else if c_is_digit == is_numeric {
            current_chunk.push(c);
        } else {
            // Transition between numeric and non-numeric
            if is_numeric {
                let num: i64 = current_chunk.parse().unwrap_or(0);
                result.push((true, num, String::new()));
            } else {
                result.push((false, 0, current_chunk.to_lowercase()));
            }
            current_chunk = c.to_string();
            is_numeric = c_is_digit;
        }
    }

    // Don't forget the last chunk
    if !current_chunk.is_empty() {
        if is_numeric {
            let num: i64 = current_chunk.parse().unwrap_or(0);
            result.push((true, num, String::new()));
        } else {
            result.push((false, 0, current_chunk.to_lowercase()));
        }
    }

    result
}

/// Compare two strings using natural sort order.
/// Numbers are compared numerically, text is compared lexicographically.
fn natural_compare(a: &str, b: &str) -> std::cmp::Ordering {
    let key_a = natural_sort_key(a);
    let key_b = natural_sort_key(b);

    for (chunk_a, chunk_b) in key_a.iter().zip(key_b.iter()) {
        let ord = match (chunk_a.0, chunk_b.0) {
            (true, true) => chunk_a.1.cmp(&chunk_b.1),
            (false, false) => chunk_a.2.cmp(&chunk_b.2),
            (true, false) => std::cmp::Ordering::Less, // Numbers before text
            (false, true) => std::cmp::Ordering::Greater,
        };
        if ord != std::cmp::Ordering::Equal {
            return ord;
        }
    }

    key_a.len().cmp(&key_b.len())
}

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

/// Load a sequence of TIFF files from a folder as a 3D volume.
/// Files are sorted using natural sort order (img_1, img_2, img_10, not img_1, img_10, img_2).
/// Only .tif and .tiff files are included; other files are silently ignored.
/// Shape will be [num_files, height, width].
pub fn load_tiff_sequence(folder: &Path) -> Result<Volume3D, DataLoadError> {
    // Read directory and collect TIFF files
    let entries = fs::read_dir(folder).map_err(|e| DataLoadError::IoError(e.to_string()))?;

    let mut tiff_paths: Vec<_> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| {
                    let lower = ext.to_lowercase();
                    lower == "tif" || lower == "tiff"
                })
                .unwrap_or(false)
        })
        .collect();

    if tiff_paths.is_empty() {
        return Err(DataLoadError::TiffError(
            "No TIFF files found in folder".to_string(),
        ));
    }

    // Sort using natural sort order
    tiff_paths.sort_by(|a, b| {
        let name_a = a.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let name_b = b.file_name().and_then(|n| n.to_str()).unwrap_or("");
        natural_compare(name_a, name_b)
    });

    // Load each TIFF file
    let mut images: Vec<Vec<f32>> = Vec::new();
    let mut width = 0usize;
    let mut height = 0usize;

    for (idx, path) in tiff_paths.iter().enumerate() {
        let file = File::open(path).map_err(|e| {
            DataLoadError::IoError(format!("Failed to open {:?}: {}", path.file_name(), e))
        })?;
        let reader = BufReader::new(file);
        let mut decoder = Decoder::new(reader).map_err(|e| {
            DataLoadError::TiffError(format!("Failed to decode {:?}: {}", path.file_name(), e))
        })?;

        let (w, h) = decoder.dimensions().map_err(|e| {
            DataLoadError::TiffError(format!(
                "Failed to get dimensions of {:?}: {}",
                path.file_name(),
                e
            ))
        })?;

        let color_type = decoder.colortype().map_err(|e| {
            DataLoadError::TiffError(format!(
                "Failed to get color type of {:?}: {}",
                path.file_name(),
                e
            ))
        })?;

        // Only support grayscale
        if !matches!(
            color_type,
            ColorType::Gray(8) | ColorType::Gray(16) | ColorType::Gray(32)
        ) {
            return Err(DataLoadError::UnsupportedDataType(format!(
                "Unsupported TIFF color type in {:?}: {:?}. Only grayscale supported.",
                path.file_name(),
                color_type
            )));
        }

        // Check dimensions consistency
        if idx == 0 {
            width = w as usize;
            height = h as usize;
        } else if w as usize != width || h as usize != height {
            return Err(DataLoadError::InvalidDimensions(format!(
                "TIFF {:?} has dimensions {}x{}, expected {}x{} (based on first file)",
                path.file_name(),
                w,
                h,
                width,
                height
            )));
        }

        let image_data = decoder.read_image().map_err(|e| {
            DataLoadError::TiffError(format!("Failed to read {:?}: {}", path.file_name(), e))
        })?;

        let image_f32: Vec<f32> = match image_data {
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

        images.push(image_f32);
    }

    let num_images = images.len();

    // Create 3D array [num_images, height, width]
    let mut data = Array3::<f32>::zeros((num_images, height, width));
    for (img_idx, img_data) in images.into_iter().enumerate() {
        for (pixel_idx, val) in img_data.into_iter().enumerate() {
            let y = pixel_idx / width;
            let x = pixel_idx % width;
            if y < height && x < width {
                data[[img_idx, y, x]] = val;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_natural_sort_simple() {
        let mut files = vec!["img_2.tif", "img_10.tif", "img_1.tif"];
        files.sort_by(|a, b| natural_compare(a, b));
        assert_eq!(files, vec!["img_1.tif", "img_2.tif", "img_10.tif"]);
    }

    #[test]
    fn test_natural_sort_complex() {
        let mut files = vec![
            "img_1.tif",
            "img_10.tif",
            "img_11.tif",
            "img_2.tif",
            "img_20.tif",
            "img_3.tif",
        ];
        files.sort_by(|a, b| natural_compare(a, b));
        assert_eq!(
            files,
            vec![
                "img_1.tif",
                "img_2.tif",
                "img_3.tif",
                "img_10.tif",
                "img_11.tif",
                "img_20.tif"
            ]
        );
    }

    #[test]
    fn test_natural_sort_zero_padded() {
        let mut files = vec!["slice_0010.tif", "slice_0001.tif", "slice_0002.tif"];
        files.sort_by(|a, b| natural_compare(a, b));
        assert_eq!(
            files,
            vec!["slice_0001.tif", "slice_0002.tif", "slice_0010.tif"]
        );
    }

    #[test]
    fn test_natural_sort_mixed_case() {
        let mut files = vec!["IMG_2.tif", "img_1.tif", "Img_10.tif"];
        files.sort_by(|a, b| natural_compare(a, b));
        assert_eq!(files, vec!["img_1.tif", "IMG_2.tif", "Img_10.tif"]);
    }
}
