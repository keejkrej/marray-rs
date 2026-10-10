use ndarray::Array2;

/// Binary erosion on a 2D binary image.
pub fn binary_erosion(image: &Array2<u8>) -> Array2<u8> {
    let (height, width) = image.dim();
    let mut out = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let mut all_one = true;
            for ny in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                for nx in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                    if image[[ny, nx]] == 0 {
                        all_one = false;
                    }
                }
            }
            out[[y, x]] = if all_one { 1 } else { 0 };
        }
    }
    out
}

/// Binary dilation on a 2D binary image.
pub fn binary_dilation(image: &Array2<u8>) -> Array2<u8> {
    let (height, width) = image.dim();
    let mut out = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let mut any_one = false;
            for ny in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                for nx in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                    if image[[ny, nx]] == 1 {
                        any_one = true;
                    }
                }
            }
            out[[y, x]] = if any_one { 1 } else { 0 };
        }
    }
    out
}

/// Binary erosion with a custom structuring element (bool version).
pub fn binary_erosion_selem(image: &Array2<bool>, selem: &Array2<bool>) -> Array2<bool> {
    let (height, width) = image.dim();
    let (sh, sw) = selem.dim();
    let cy = sh / 2;
    let cx = sw / 2;
    let mut out = Array2::from_elem((height, width), false);
    for y in 0..height {
        for x in 0..width {
            let mut all_true = true;
            for sy in 0..sh {
                for sx in 0..sw {
                    if !selem[[sy, sx]] {
                        continue;
                    }
                    let ny = y as isize + sy as isize - cy as isize;
                    let nx = x as isize + sx as isize - cx as isize;
                    if ny < 0 || ny >= height as isize || nx < 0 || nx >= width as isize {
                        all_true = false;
                        break;
                    }
                    if !image[[ny as usize, nx as usize]] {
                        all_true = false;
                        break;
                    }
                }
                if !all_true {
                    break;
                }
            }
            out[[y, x]] = all_true;
        }
    }
    out
}

/// Binary dilation with a custom structuring element (bool version).
pub fn binary_dilation_selem(image: &Array2<bool>, selem: &Array2<bool>) -> Array2<bool> {
    let (height, width) = image.dim();
    let (sh, sw) = selem.dim();
    let cy = sh / 2;
    let cx = sw / 2;
    let mut out = Array2::from_elem((height, width), false);
    for y in 0..height {
        for x in 0..width {
            let mut any_true = false;
            for sy in 0..sh {
                for sx in 0..sw {
                    if !selem[[sy, sx]] {
                        continue;
                    }
                    let ny = y as isize + sy as isize - cy as isize;
                    let nx = x as isize + sx as isize - cx as isize;
                    if ny < 0 || ny >= height as isize || nx < 0 || nx >= width as isize {
                        continue;
                    }
                    if image[[ny as usize, nx as usize]] {
                        any_true = true;
                        break;
                    }
                }
                if any_true {
                    break;
                }
            }
            out[[y, x]] = any_true;
        }
    }
    out
}

/// Binary opening (bool): erosion then dilation with structuring element.
pub fn binary_opening(image: &Array2<bool>, selem: &Array2<bool>) -> Array2<bool> {
    binary_dilation_selem(&binary_erosion_selem(image, selem), selem)
}

/// Binary closing (bool): dilation then erosion with structuring element.
pub fn binary_closing(image: &Array2<bool>, selem: &Array2<bool>) -> Array2<bool> {
    binary_erosion_selem(&binary_dilation_selem(image, selem), selem)
}

/// Grayscale erosion (min filter with structuring element).
pub fn erosion(image: &Array2<f64>, selem: &Array2<f64>) -> Array2<f64> {
    let (height, width) = image.dim();
    let (sh, sw) = selem.dim();
    let cy = sh / 2;
    let cx = sw / 2;
    let mut out = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let mut min_val = f64::INFINITY;
            for sy in 0..sh {
                for sx in 0..sw {
                    if selem[[sy, sx]] == 0.0 {
                        continue;
                    }
                    let ny = y as isize + sy as isize - cy as isize;
                    let nx = x as isize + sx as isize - cx as isize;
                    if ny < 0 || ny >= height as isize || nx < 0 || nx >= width as isize {
                        continue;
                    }
                    let val = image[[ny as usize, nx as usize]] - selem[[sy, sx]];
                    if val < min_val {
                        min_val = val;
                    }
                }
            }
            out[[y, x]] = if min_val == f64::INFINITY { 0.0 } else { min_val };
        }
    }
    out
}

/// Grayscale dilation (max filter with structuring element).
pub fn dilation(image: &Array2<f64>, selem: &Array2<f64>) -> Array2<f64> {
    let (height, width) = image.dim();
    let (sh, sw) = selem.dim();
    let cy = sh / 2;
    let cx = sw / 2;
    let mut out = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let mut max_val = f64::NEG_INFINITY;
            for sy in 0..sh {
                for sx in 0..sw {
                    if selem[[sy, sx]] == 0.0 {
                        continue;
                    }
                    let ny = y as isize + sy as isize - cy as isize;
                    let nx = x as isize + sx as isize - cx as isize;
                    if ny < 0 || ny >= height as isize || nx < 0 || nx >= width as isize {
                        continue;
                    }
                    let val = image[[ny as usize, nx as usize]] + selem[[sy, sx]];
                    if val > max_val {
                        max_val = val;
                    }
                }
            }
            out[[y, x]] = if max_val == f64::NEG_INFINITY { 0.0 } else { max_val };
        }
    }
    out
}

/// Grayscale opening: erosion then dilation.
pub fn opening(image: &Array2<f64>, selem: &Array2<f64>) -> Array2<f64> {
    dilation(&erosion(image, selem), selem)
}

/// Grayscale closing: dilation then erosion.
pub fn closing(image: &Array2<f64>, selem: &Array2<f64>) -> Array2<f64> {
    erosion(&dilation(image, selem), selem)
}

/// Zhang-Suen thinning algorithm for skeletonization.
pub fn skeletonize(image: &Array2<bool>) -> Array2<bool> {
    let (height, width) = image.dim();
    let mut skel = image.clone();
    loop {
        let mut to_remove = Vec::new();
        // Step 1
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                if !skel[[y, x]] {
                    continue;
                }
                let p = [
                    skel[[y - 1, x]],     // p2
                    skel[[y - 1, x + 1]], // p3
                    skel[[y, x + 1]],     // p4
                    skel[[y + 1, x + 1]], // p5
                    skel[[y + 1, x]],     // p6
                    skel[[y + 1, x - 1]], // p7
                    skel[[y, x - 1]],     // p8
                    skel[[y - 1, x - 1]], // p9
                ];
                let bp = p.iter().filter(|&&v| v).count();
                if !(2..=6).contains(&bp) {
                    continue;
                }
                let ap = transitions(&p);
                if ap != 1 {
                    continue;
                }
                if !(!p[0] || !p[2] || !p[4]) {
                    continue;
                }
                if !(!p[2] || !p[4] || !p[6]) {
                    continue;
                }
                to_remove.push((y, x));
            }
        }
        for &(y, x) in &to_remove {
            skel[[y, x]] = false;
        }
        // Step 2
        to_remove.clear();
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                if !skel[[y, x]] {
                    continue;
                }
                let p = [
                    skel[[y - 1, x]],     // p2
                    skel[[y - 1, x + 1]], // p3
                    skel[[y, x + 1]],     // p4
                    skel[[y + 1, x + 1]], // p5
                    skel[[y + 1, x]],     // p6
                    skel[[y + 1, x - 1]], // p7
                    skel[[y, x - 1]],     // p8
                    skel[[y - 1, x - 1]], // p9
                ];
                let bp = p.iter().filter(|&&v| v).count();
                if !(2..=6).contains(&bp) {
                    continue;
                }
                let ap = transitions(&p);
                if ap != 1 {
                    continue;
                }
                if !(!p[0] || !p[2] || !p[6]) {
                    continue;
                }
                if !(!p[0] || !p[4] || !p[6]) {
                    continue;
                }
                to_remove.push((y, x));
            }
        }
        if to_remove.is_empty() {
            break;
        }
        for &(y, x) in &to_remove {
            skel[[y, x]] = false;
        }
    }
    skel
}

fn transitions(p: &[bool; 8]) -> usize {
    let mut count = 0;
    for i in 0..8 {
        if p[i] && !p[(i + 1) % 8] {
            count += 1;
        }
    }
    count
}

/// Remove labeled regions smaller than min_size.
pub fn remove_small_objects(labels: &Array2<usize>, min_size: usize) -> Array2<usize> {
    let (height, width) = labels.dim();
    let max_label = labels.iter().copied().max().unwrap_or(0);
    let mut counts = vec![0usize; max_label + 1];
    for &v in labels.iter() {
        counts[v] += 1;
    }
    let mut out = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let v = labels[[y, x]];
            if v > 0 && counts[v] >= min_size {
                out[[y, x]] = v;
            }
        }
    }
    out
}

/// Create a disk (circle) structuring element of given radius.
pub fn disk(radius: usize) -> Array2<bool> {
    let size = 2 * radius + 1;
    let mut selem = Array2::from_elem((size, size), false);
    let r = radius as f64;
    for y in 0..size {
        for x in 0..size {
            let dy = y as f64 - radius as f64;
            let dx = x as f64 - radius as f64;
            if (dy * dy + dx * dx).sqrt() <= r + 0.5 {
                selem[[y, x]] = true;
            }
        }
    }
    selem
}

/// Create a square structuring element of given size.
pub fn square(size: usize) -> Array2<bool> {
    Array2::from_elem((size, size), true)
}

/// Create a diamond structuring element of given radius.
pub fn diamond(radius: usize) -> Array2<bool> {
    let size = 2 * radius + 1;
    let mut selem = Array2::from_elem((size, size), false);
    for y in 0..size {
        for x in 0..size {
            let dy = (y as isize - radius as isize).unsigned_abs();
            let dx = (x as isize - radius as isize).unsigned_abs();
            if dy + dx <= radius {
                selem[[y, x]] = true;
            }
        }
    }
    selem
}

/// Connected component labeling (delegates to sp::ndimage::label pattern).
pub fn label(image: &Array2<bool>) -> (Array2<usize>, usize) {
    let (height, width) = image.dim();
    let mut labels = Array2::zeros((height, width));
    let mut current_label = 0usize;
    for y in 0..height {
        for x in 0..width {
            if image[[y, x]] && labels[[y, x]] == 0 {
                current_label += 1;
                let mut stack = vec![(y, x)];
                while let Some((r, c)) = stack.pop() {
                    if r >= height || c >= width || !image[[r, c]] || labels[[r, c]] != 0 {
                        continue;
                    }
                    labels[[r, c]] = current_label;
                    if r > 0 {
                        stack.push((r - 1, c));
                    }
                    if r + 1 < height {
                        stack.push((r + 1, c));
                    }
                    if c > 0 {
                        stack.push((r, c - 1));
                    }
                    if c + 1 < width {
                        stack.push((r, c + 1));
                    }
                }
            }
        }
    }
    (labels, current_label)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_ops() {
        let image = Array2::from_shape_vec((3, 3), vec![1, 1, 1, 1, 1, 1, 1, 1, 1]).unwrap();
        let eroded = binary_erosion(&image);
        let dilated = binary_dilation(&image);
        assert_eq!(eroded.dim(), (3, 3));
        assert_eq!(dilated.dim(), (3, 3));
        assert_eq!(eroded[[1, 1]], 1);
        assert_eq!(dilated[[1, 1]], 1);
    }

    #[test]
    fn test_binary_opening_closing() {
        let mut image = Array2::from_elem((8, 8), false);
        // 8x8 binary circle-ish shape
        for y in 2..6 {
            for x in 2..6 {
                image[[y, x]] = true;
            }
        }
        let selem = square(3);
        let opened = binary_opening(&image, &selem);
        let closed = binary_closing(&image, &selem);
        assert_eq!(opened.dim(), (8, 8));
        assert_eq!(closed.dim(), (8, 8));
        // Opening should preserve the interior
        assert!(opened[[3, 3]]);
        // Closing should fill holes (no holes here, so same shape)
        assert!(closed[[3, 3]]);
    }

    #[test]
    fn test_binary_opening_removes_noise() {
        let mut image = Array2::from_elem((8, 8), false);
        // large block
        for y in 3..6 {
            for x in 3..6 {
                image[[y, x]] = true;
            }
        }
        // single-pixel noise
        image[[0, 0]] = true;
        let selem = square(3);
        let opened = binary_opening(&image, &selem);
        // noise should be removed
        assert!(!opened[[0, 0]]);
        // block interior preserved
        assert!(opened[[4, 4]]);
    }

    #[test]
    fn test_binary_closing_fills_gaps() {
        let mut image = Array2::from_elem((8, 8), false);
        // block with hole in center
        for y in 2..6 {
            for x in 2..6 {
                image[[y, x]] = true;
            }
        }
        image[[4, 4]] = false; // hole
        let selem = square(3);
        let closed = binary_closing(&image, &selem);
        // hole should be filled
        assert!(closed[[4, 4]]);
    }

    #[test]
    fn test_grayscale_erosion_dilation() {
        let image = Array2::from_shape_vec(
            (3, 3),
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        )
        .unwrap();
        let selem = Array2::from_elem((3, 3), 1.0);
        let eroded = erosion(&image, &selem);
        let dilated = dilation(&image, &selem);
        // center pixel should be reduced by erosion
        assert!(eroded[[1, 1]] < 1.0);
        // dilation should spread the peak
        assert!(dilated[[1, 0]] > 0.0);
    }

    #[test]
    fn test_grayscale_opening_closing() {
        let image = Array2::from_shape_vec(
            (3, 3),
            vec![0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.5],
        )
        .unwrap();
        let selem = Array2::from_elem((3, 3), 1.0);
        let opened = opening(&image, &selem);
        let closed = closing(&image, &selem);
        assert_eq!(opened.dim(), (3, 3));
        assert_eq!(closed.dim(), (3, 3));
        // opening should not increase values
        assert!(opened[[1, 1]] <= image[[1, 1]]);
        // closing should not decrease values
        assert!(closed[[1, 1]] >= image[[1, 1]]);
    }

    #[test]
    fn test_skeletonize() {
        // horizontal bar 3 pixels tall, 8 pixels wide
        let mut image = Array2::from_elem((8, 8), false);
        for y in 3..6 {
            for x in 1..7 {
                image[[y, x]] = true;
            }
        }
        let skel = skeletonize(&image);
        assert_eq!(skel.dim(), (8, 8));
        // skeleton should be thinner - center row should have pixels
        assert!(skel[[4, 3]]);
        // top/bottom rows of the bar should be eroded
        // (depending on shape, at least some should be false)
        let top_count: usize = (1..7).filter(|&x| skel[[3, x]]).count();
        assert!(top_count < 6);
    }

    #[test]
    fn test_skeletonize_cross() {
        let mut image = Array2::from_elem((5, 5), false);
        // cross shape
        image[[2, 1]] = true;
        image[[2, 2]] = true;
        image[[2, 3]] = true;
        image[[1, 2]] = true;
        image[[3, 2]] = true;
        let skel = skeletonize(&image);
        // center should remain
        assert!(skel[[2, 2]]);
    }

    #[test]
    fn test_remove_small_objects() {
        let mut labels = Array2::zeros((8, 8));
        // region 1: 5 pixels
        for y in 0..2 {
            for x in 0..3 {
                labels[[y, x]] = 1;
            }
        }
        // region 2: 2 pixels (small)
        labels[[5, 5]] = 2;
        labels[[5, 6]] = 2;
        let result = remove_small_objects(&labels, 3);
        // region 1 kept
        assert_eq!(result[[0, 0]], 1);
        // region 2 removed
        assert_eq!(result[[5, 5]], 0);
    }

    #[test]
    fn test_disk() {
        let d = disk(2);
        assert_eq!(d.dim(), (5, 5));
        // center should be true
        assert!(d[[2, 2]]);
        // corners should be false (distance > radius)
        assert!(!d[[0, 0]]);
        // cardinal directions should be true
        assert!(d[[0, 2]]);
        assert!(d[[2, 0]]);
        assert!(d[[4, 2]]);
        assert!(d[[2, 4]]);
    }

    #[test]
    fn test_square() {
        let s = square(4);
        assert_eq!(s.dim(), (4, 4));
        assert!(s.iter().all(|&v| v));
    }

    #[test]
    fn test_diamond() {
        let d = diamond(2);
        assert_eq!(d.dim(), (5, 5));
        assert!(d[[2, 2]]);
        assert!(d[[0, 2]]);
        assert!(d[[2, 0]]);
        // corners false
        assert!(!d[[0, 0]]);
        assert!(!d[[0, 4]]);
    }

    #[test]
    fn test_label() {
        let mut image = Array2::from_elem((8, 8), false);
        // two separate regions
        image[[1, 1]] = true;
        image[[1, 2]] = true;
        image[[6, 6]] = true;
        let (labels, n) = label(&image);
        assert_eq!(n, 2);
        assert_eq!(labels[[1, 1]], labels[[1, 2]]);
        assert_ne!(labels[[1, 1]], labels[[6, 6]]);
    }
}
