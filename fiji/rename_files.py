#@ File	(label = "Roi input directory", style = "directory") roiFile
#@ File	(label = "Fluorescence directory", style = "directory") imFile
#@ File	(label = "Output directory", style = "directory") dstFile
#@ String  (label = "File name contains", value = "") containString
#@ String (label="Attach string to Roi-file name", value="Hole-Roi_") roiAdd
#@ String  (label = "File extension", value=".tif") ext
#@ Integer (label="Position of Well Information in image title") posImg
#@ Integer (label="Position of Well Information in roi title") posRoi

import os, glob, shutil



roiFile = roiFile.getAbsolutePath()
imFile = imFile.getAbsolutePath()
dstFile = dstFile.getAbsolutePath()

roiFileList = glob.glob(os.path.join(roiFile, "*.roi"))

if not "." in ext:
	ext = "." + ext
imFileList = glob.glob(os.path.join(imFile, "*%s"%ext))

fileSorter = {}

for r in imFileList:
	if containString in r:
		f = os.path.split(r)[1]
		well = f.split("_")[posImg]
		well, ext1 = os.path.splitext(well)
		fileSorter[well] = [well]
	
print(imFileList)

	
for r in roiFileList:
	if containString in r:
		f = os.path.split(r)[1]
		print(f)
		well = f.split("_")[posRoi]
		well, ext1 = os.path.splitext(well)
		fileSorter[well].append(r)
		#if well in fileSorter:
		#	fileSorter[well].append(r)
		#else:
		#	print("Warning: unexpected ROI file '%s'. Skipping." % well)
			
print(roiFileList)
	
for k,v in fileSorter.items():
	print(k, ": ", v)

	try:
		for items in v:
			if os.path.splitext(items)[1] == ".roi":
				destPath = os.path.join(dstFile, os.path.split(items)[1])
				print(destPath)
				if not os.path.isfile(destPath):
					shutil.copy(items, destPath)
				newName = destPath.replace(destPath.split("_")[-1], v[0].split("_")[-1])
				newName = newName.replace(ext, ".roi")
				newName = os.path.join(os.path.split(newName)[0], roiAdd + os.path.split(newName)[1])
				print(newName)
				os.rename(destPath, newName)
	except:
		continue

		
	
	