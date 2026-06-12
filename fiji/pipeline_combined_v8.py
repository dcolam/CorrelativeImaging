#@ File    (label="Input image directory", style="directory") input_folder
#@ File    (label="Output directory", style="directory") output_folder
#@ File    (label="Select a trained Ilastik-Project", style="file") ilastikPath
#@ String  (label="Image extension", value=".vsi") extension
#@ String  (label="File name contains", value="") containString
#@ String  (label="Brightfield suffix", value="00001") bfSuffix
#@ String  (label="Fluorescence suffix", value="00002") flSuffix
#@ Integer (label="Number of Ilastik classes", value=3) numClass
#@ Integer (label="Crop radius (pixels)", value=100) crop_radius
#@ String  (visibility=MESSAGE, value="--- Brightfield crop slice ---") msgBF
#@ Integer (label="BF slice for crop (0 = auto-detect sharpest)", value=0) bfSliceOverride
#@ ColorRGB (label="Brightfield color", value="white") bfColor
#@ String  (visibility=MESSAGE, value="--- Fluorescence Z-Projection types ---") msg1
#@ Boolean (label="Average Projection",  value=false) avg
#@ Boolean (label="Minimum Projection",  value=false) minBool
#@ Boolean (label="Maximum Projection",  value=true)  maxBool
#@ Boolean (label="Sum Projection",      value=false) sumBool
#@ Boolean (label="Sd Projection",       value=false) sd
#@ Boolean (label="Median Projection",   value=false) medianBool
#@ String  (visibility=MESSAGE, value="--- Fluorescence channel colors ---") msg2
#@ ColorRGB (label="Channel 1 color", value="red")     ch1Color
#@ ColorRGB (label="Channel 2 color", value="green")   ch2Color
#@ ColorRGB (label="Channel 3 color", value="blue")    ch3Color
#@ ColorRGB (label="Channel 4 color", value="magenta") ch4Color
#@ String  (visibility=MESSAGE, value="--- Fluorescence display stretch ---") msg3
#@ Float   (label="Stretch saturation % (0=min/max, 0.35=default)", value=1.0, min=0.0, max=5.0, stepSize=0.05) stretchSaturation
#@ Integer (label="Max upper display value per channel (0 = no limit)", value=0, min=0, max=65535) stretchMaxVal

# ============================================================
#  PIPELINE — per well (_00001 BF / _00002 fluorescence):
#
#  BF stack  ──► minimum projection  ──► Ilastik ──► best ROI
#            └─► sharpest slice (or manual slice) ──► crop BF
#
#  Fluorescence stack ──► Z-projection (user choice) ──► crop
#                         same ROI applied to both crops
#
#  Outputs per Ilastik class:
#    <well>_BF_classN_crop.tif        (raw, chosen BF slice, with BF color LUT)
#    <well>_BF_classN_crop.jpg        (contrast-stretched + scale bar)
#    <well>_fluoro_classN_crop.tif    (raw multi-channel)
#    <well>_fluoro_classN_crop.jpg    (colored + stretched + scale bar)
# ============================================================

import os, re, math
from ij import IJ, ImagePlus
from ij.plugin import ZProjector as zp, ImagesToStack, ContrastEnhancer
from ij.plugin.frame import RoiManager
from ij.gui import Overlay, ShapeRoi
from ij.process import ImageConverter, ImageStatistics, LUT
from loci.plugins import BF
from loci.plugins.in import ImporterOptions
from java.io import File
from java.awt import Color

fl_colors  = [ch1Color, ch2Color, ch3Color, ch4Color]
fl_types_bool = [avg, minBool, maxBool, sumBool, sd, medianBool]
fl_type_names = ["avg", "min", "max", "sum", "sd", "median"]
fl_selected   = [t for t, b in zip(fl_type_names, fl_types_bool) if b]
if not fl_selected:
    fl_selected = ["max"]

# ------------------------------------------------------------------ helpers

def open_image(path):
    options = ImporterOptions()
    options.setColorMode(ImporterOptions.COLOR_MODE_COLORIZED)
    options.setAutoscale(True) 
    options.setId(path)
    return BF.openImagePlus(options)[0]


def z_project(imp, types):
    """Z-project imp using each method in types list; stack results if >1."""
    projector = zp(imp)
    results   = [projector.run(imp, "%s all" % t) for t in types]
    if len(results) == 1:
        return results[0]
    return ImagesToStack.run(results)


def slice_variance(ip):
    """
    Laplacian-based focus measure on a single ImageProcessor.
    Higher = sharper. Uses the sum of squared differences between
    adjacent pixels (approximates Laplacian energy).
    """
    arr   = ip.convertToFloat().getFloatArray()
    h     = len(arr)
    w     = len(arr[0]) if h > 0 else 0
    total = 0.0
    count = 0
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            lap = (4 * arr[y][x]
                   - arr[y-1][x] - arr[y+1][x]
                   - arr[y][x-1] - arr[y][x+1])
            total += lap * lap
            count += 1
    return total / count if count > 0 else 0.0


def find_sharpest_slice(imp):
    """
    Iterate all slices of imp (channel 1 if multichannel) and return
    the 1-based index of the slice with the highest Laplacian energy.
    Prints a score table so the user can verify / pick manually next time.
    """
    n = imp.getNSlices()
    if n <= 1:
        print("  BF has only 1 slice — using it.")
        return 1

    scores = []
    orig_slice = imp.getSlice()
    for z in range(1, n + 1):
        imp.setPositionWithoutUpdate(1, z, 1)   # channel=1, slice=z, frame=1
        ip    = imp.getProcessor().duplicate()
        score = slice_variance(ip)
        scores.append(score)
        print("    Slice %2d: focus score = %.2f" % (z, score))

    imp.setSlice(orig_slice)
    best_idx = scores.index(max(scores)) + 1   # 1-based
    print("  → Auto-selected sharpest slice: %d" % best_idx)
    return best_idx


def make_lut_from_color(r, g, b):
    """
    Build a 256-entry LUT that ramps from black to (r, g, b).
    LUT constructor requires Java byte[] arrays — Python values must be
    cast to signed bytes (Java bytes are signed: 128..255 → -128..-1).
    """
    from jarray import array as jarray
    reds   = jarray([int(r * i / 255) if int(r * i / 255) < 128
                     else int(r * i / 255) - 256 for i in range(256)], 'b')
    greens = jarray([int(g * i / 255) if int(g * i / 255) < 128
                     else int(g * i / 255) - 256 for i in range(256)], 'b')
    blues  = jarray([int(b * i / 255) if int(b * i / 255) < 128
                     else int(b * i / 255) - 256 for i in range(256)], 'b')
    return LUT(reds, greens, blues)


def extract_slice_as_grey(imp, slice_idx):
    """
    Extract slice_idx (1-based) from imp as a new greyscale 8-bit ImagePlus
    with the user-chosen BF color LUT applied.
    """
    imp.setSlice(slice_idx)
    ip     = imp.getProcessor().duplicate()
    result = ImagePlus(imp.getTitle(), ip)
    ImageConverter(result).convertToGray8()
    lut = make_lut_from_color(bfColor.getRed(), bfColor.getGreen(), bfColor.getBlue())
    result.getProcessor().setLut(lut)
    return result


def circularity(roi):
    """4*pi*area / perimeter^2  —  1.0 = perfect circle."""
    area  = roi.getStatistics().area
    perim = roi.getLength()
    if perim == 0:
        return 0.0
    return (4.0 * math.pi * area) / (perim * perim)


def best_sub_roi(roi):
    """
    From a compound ROI keep the part with the best combined score:
    normalised_area * circularity.  Falls back if only one part.
    """
    shape = ShapeRoi(roi)
    parts = list(shape.getRois())
    if len(parts) <= 1:
        return roi
    areas    = [r.getStatistics().area for r in parts]
    max_area = max(areas) if max(areas) > 0 else 1.0
    scores   = [(areas[i] / max_area) * circularity(parts[i])
                for i in range(len(parts))]
    best_idx = scores.index(max(scores))
    print("    ROI: %d parts — keeping part %d  (score=%.3f, area=%.0f, circ=%.3f)"
          % (len(parts), best_idx,
             scores[best_idx], areas[best_idx], circularity(parts[best_idx])))
    return parts[best_idx]


def run_ilastik(imp):
    """Run Ilastik pixel classification; return list of per-class ROIs."""
    project = ilastikPath.getAbsolutePath()
    args = ("projectfilename=%s inputimage=[%s] "
            "pixelclassificationtype=Segmentation" % (project, imp.getTitle()))
    imp.show()
    IJ.run("Run Pixel Classification Prediction", args)
    imp.hide()
    pred_imp = IJ.getImage()

    rois = []
    for cls in range(1, numClass + 1):
        IJ.run(pred_imp, "Manual Threshold...", "min=%s max=%s" % (cls, cls))
        IJ.run(pred_imp, "Create Selection", "")
        roi = pred_imp.getRoi()
        if roi is not None:
            rois.append(roi)
        else:
            print("  Warning: Ilastik class %d produced no ROI." % cls)

    pred_imp.close()
    return rois


def apply_fl_luts(imp):
    """Apply user-defined fluorescence channel LUTs."""
    n = min(imp.getNChannels(), 4)
    for c in range(1, n + 1):
    	print(c)
        col = fl_colors[c - 1]
        print(col)
        lut = make_lut_from_color(col.getRed(), col.getGreen(), col.getBlue())
        imp.setChannelLut(lut, c)


def save_roi(imp, roi, roi_path):
    """Save ROI using IJ.saveAs Selection — same method as original pipeline."""
    roi.setName(os.path.basename(roi_path).replace(".roi", ""))
    imp.setRoi(roi)
    IJ.saveAs(imp, "Selection", roi_path)
    print("    Saved ROI: %s" % os.path.basename(roi_path))


def stretch_fl_for_display(imp):
    """
    Non-destructively stretch all fluorescence channels for display.
    Returns a NEW duplicate with LUTs applied and contrast stretched —
    the original imp is never modified.

    stretchSaturation: % of pixels clipped at each end (0 = true min/max)
    stretchMaxVal:     if > 0, clamps the upper display limit to this grey
                       value after stretching (useful to suppress hot pixels
                       or background that dominates the histogram top end).
    """
    stretched = imp.duplicate()
    apply_fl_luts(stretched)
    stretched.resetDisplayRanges()
    stretched.setDisplayMode(IJ.COLOR)
    ce = ContrastEnhancer()
    for c in range(1, stretched.getNChannels() + 1):
        stretched.setC(c)
        ip = stretched.getProcessor()
        ce.stretchHistogram(stretched, stretchSaturation)
        #stretchedMax = stretched.getMax()
        stretchedMax = 0
        if stretchMaxVal > stretchedMax:
            # Keep the auto-detected min but cap the max
            current_min = 0
            print("capping channel%s from %s to max value %s"%(c,stretchedMax, stretchMaxVal))
            ip.setMinAndMax(ip.getMin(), stretchMaxVal)
        #stretched.setProcessor(ip)
    stretched.setDisplayMode(IJ.COMPOSITE)
    return stretched


def crop_and_save(display_imp, roi, out_dir, out_base, is_fluorescence, raw_imp=None):
    """
    Save a raw .tif crop and a display .jpg crop centred on roi.
    - raw_imp:     the unmodified image — used for the .tif (defaults to display_imp)
    - display_imp: already has LUTs applied and contrast stretched — used for .jpg
    No stretching happens here; all display preparation must be done before calling.
    """
    src_raw = raw_imp if raw_imp is not None else display_imp
    bounds = roi.getBounds()
    cx = bounds.x + bounds.width  / 2
    cy = bounds.y + bounds.height / 2
    x  = int(cx - crop_radius)
    y  = int(cy - crop_radius)
    w  = h = crop_radius * 2

    x = max(0, min(x, src_raw.getWidth()  - w))
    y = max(0, min(y, src_raw.getHeight() - h))

    # --- raw TIF from unmodified image ---
    src_raw.setRoi(x, y, w, h)
    raw_crop = src_raw.crop("stack")
    IJ.saveAsTiff(raw_crop, os.path.join(out_dir, out_base + ".tif"))
    raw_crop.close()

    # --- JPG from display copy (already stretched + LUTs) ---
    display_imp.setOverlay(Overlay(roi))
    display_imp.setRoi(x, y, w, h)
    jpg_crop = display_imp.crop("stack")
    flat = jpg_crop.flatten()
    IJ.run(flat, "Scale Bar...",
           "width=10 height=10 font=18 color=White background=None "
           "location=[Lower Left] horizontal bold overlay")
    flat = flat.flatten()
    IJ.saveAs(flat, "jpg", os.path.join(out_dir, out_base))
    flat.close()
    jpg_crop.close()


# ------------------------------------------------------------------ main

def process_well(well_name, bf_path, fl_path, out_dir, roi_dir):
    print("\n=== Well: %s ===" % well_name)

    # --- 1. Open BF stack (kept open for later slice extraction) ---
    print("  Opening BF:", os.path.basename(bf_path))
    bf_imp = open_image(bf_path)

    # --- 2. Minimum projection → Ilastik ---
    print("  Creating minimum projection for Ilastik...")
    bf_min = z_project(bf_imp, ["min"])
    bf_min.setTitle("BF_min_" + well_name)

    print("  Running Ilastik segmentation...")
    raw_rois = run_ilastik(bf_min)
    bf_min.close()

    if not raw_rois:
        print("  No ROIs produced — skipping well.")
        bf_imp.close()
        return

    # --- 3. Pick BF slice for crop ---
    if bfSliceOverride > 0:
        bf_slice_idx = bfSliceOverride
        print("  Using manually specified BF slice: %d" % bf_slice_idx)
    else:
        print("  Auto-detecting sharpest BF slice...")
        #bf_slice_idx = find_sharpest_slice(bf_imp)
        bf_slice_idx = int(round((bf_imp.getNSlices()+1)/2))
        print(bf_slice_idx)

    bf_crop_imp = extract_slice_as_grey(bf_imp, bf_slice_idx)
    bf_crop_imp.setTitle("BF_crop_" + well_name)
    bf_imp.close()

    # --- 4. Open & Z-project fluorescence ---
    print("  Opening fluorescence:", os.path.basename(fl_path))
    fl_imp  = open_image(fl_path)
    fl_proj = z_project(fl_imp, fl_selected)
    fl_proj.setTitle("FL_" + well_name)
    fl_imp.close()

    # --- 5. Prepare stretched fluorescence display copy (once, before loop) ---
    fl_display = stretch_fl_for_display(fl_proj)
    fl_display.setTitle("FL_display_" + well_name)

    # --- 6. Per Ilastik class: refine ROI, save ROI, overview + crops ---
    fl_basename = os.path.splitext(os.path.basename(fl_path))[0]
    for cls_idx, raw_roi in enumerate(raw_rois):
        cls_label = "class%d" % (cls_idx + 1)
        print("  Processing %s..." % cls_label)
        roi = best_sub_roi(raw_roi)

        # Save ROI
        roi_path = os.path.join(roi_dir, "%s_%s.roi" % (cls_label, fl_basename))
        save_roi(bf_crop_imp, roi, roi_path)

        # Overview JPGs (full image, ROI overlay, scale bar)
        # BF: raw bf_crop_imp
        bf_crop_imp.setOverlay(Overlay(roi))
        bf_flat = bf_crop_imp.flatten()
        IJ.run(bf_flat, "Scale Bar...",
               "width=10 height=10 font=18 color=White background=None "
               "location=[Lower Left] horizontal bold overlay")
        bf_flat = bf_flat.flatten()
        IJ.saveAs(bf_flat, "jpg",
                  os.path.join(out_dir, "%s_BF_%s-overview_crop" % (well_name, cls_label)))
        bf_flat.close()

        # FL: stretched fl_display
        fl_display.setOverlay(Overlay(roi))
        fl_flat = fl_display.flatten()
        IJ.run(fl_flat, "Scale Bar...",
               "width=10 height=10 font=18 color=White background=None "
               "location=[Lower Left] horizontal bold overlay")
        fl_flat = fl_flat.flatten()
        IJ.saveAs(fl_flat, "jpg",
                  os.path.join(out_dir, "%s_fluoro_%s-overview_crop" % (well_name, cls_label)))
        fl_flat.close()

        # Crops — raw TIF from originals, JPG from display copies
        crop_and_save(bf_crop_imp, roi, out_dir,
                      "%s_BF_%s_crop" % (well_name, cls_label),
                      is_fluorescence=False)
        crop_and_save(fl_display, roi, out_dir,
                      "%s_fluoro_%s_crop" % (well_name, cls_label),
                      is_fluorescence=True,
                      raw_imp=fl_proj)

    fl_display.close()
    bf_crop_imp.close()
    fl_proj.close()
    print("  Done.")


def main():
    img_dir = input_folder.getAbsolutePath()
    out_dir = output_folder.getAbsolutePath()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    roi_dir = os.path.join(img_dir, "roi_folder")
    if not os.path.exists(roi_dir):
        os.makedirs(roi_dir)
    print("ROI folder: %s" % roi_dir)

    all_files = sorted([
        f for f in os.listdir(img_dir)
        if f.endswith(extension) and containString in f
    ])

    bf_map = {}
    fl_map = {}
    suffix_pattern = re.compile(
        r"_(%s|%s)%s$" % (re.escape(bfSuffix), re.escape(flSuffix),
                           re.escape(extension))
    )

    for fname in all_files:
        fpath = os.path.join(img_dir, fname)
        m = suffix_pattern.search(fname)
        if not m:
            print("Skipping (unrecognised suffix): %s" % fname)
            continue
        suffix   = m.group(1)
        well_key = fname[:m.start()]
        if suffix == bfSuffix:
            bf_map[well_key] = fpath
        elif suffix == flSuffix:
            fl_map[well_key] = fpath

    all_keys = sorted(set(bf_map.keys()) | set(fl_map.keys()))
    print("Found %d well(s)." % len(all_keys))

    failed = []
    for well_key in all_keys:
        if well_key not in bf_map:
            print("No BF image for well '%s' — skipping." % well_key)
            continue
        if well_key not in fl_map:
            print("No fluorescence image for well '%s' — skipping." % well_key)
            continue
        try:
            process_well(well_key, bf_map[well_key], fl_map[well_key], out_dir, roi_dir)
        except Exception as e:
            print("FAILED well '%s': %s" % (well_key, str(e)))
            failed.append(well_key)

    print("\n=== PIPELINE COMPLETE ===")
    if failed:
        print("Failed wells (%d):" % len(failed))
        for f in failed:
            print("  " + f)
    else:
        print("All wells processed successfully.")

main()
