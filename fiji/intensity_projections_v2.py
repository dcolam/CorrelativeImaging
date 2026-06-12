#@ File	(label = "Input directory", style = "directory") srcFile
#@ File	(label = "Output directory", style = "directory") dstFile
#@ String  (label = "File extension", value=".tif") ext
#@ String  (label = "File name contains", value = "") containString
#@ boolean (label = "Keep directory structure when saving", value = true) keepDirectories
#@ File (label= "Select a trained Ilastik-Project", style="file") ilastikPath
#@ Integer (label="Number of classes", value=3) numClass
#@ String (visibility=MESSAGE, value="Select type of projection (multiple possible)") msg
#@ Boolean (label = "Average Projection", value = true) avg
#@ Boolean (label = "Minimum Projection", value = True) minBool
#@ boolean (label = "Maximum Projection", value = true) maxBool
#@ boolean (label = "Sum Projection", value = true) sumBool
#@ boolean (label = "Sd Projection", value = true) sd
#@ boolean (label = "Median Projection", value = true) medianBool

# See also Process_Folder.ijm for a version of this code
# in the ImageJ 1.x macro language.

### (visibility=MESSAGE, value="\nSelect type of projection (multiple possible)", choices={"Average", "Minimum", "Maximum", "Sum", "Sd", "Median"}, multiple=true, value="Minimum", style="radioButtonHorizontal") types



import os, time
from ij import IJ, ImagePlus
from ij.plugin import ZProjector as zp
from loci.plugins import BF
from ij.plugin import ImagesToStack


def formatTime(start):
   t = time.time() - start
   u = " seconds"
   if t > 60:
	 t /= 60
	 u = " minutes"
	 if t > 60:
		t /= 60
		u = " hours"
		if t > 24:
		   t /= 24
		   u = " days"
   return str(round(t, 2)) + u

def zStackIJ(imp):#, types = ["avg", "min", "max", "sum", "sd", "median"]):
	types_bool = [avg, minBool, maxBool, sumBool, sd, medianBool]
	types = ["avg", "min", "max", "sum", "sd", "median"]
	types = [t for t,b in zip(types, types_bool) if b]
	z = zp(imp)
	ips = [z.run(imp,"%s all"%t) for t in types]
	if len(ips) == 1:
		return ips[0]
	else:
		return ImagesToStack.run(ips)

def run():
	start = time.time()
	srcDir = srcFile.getAbsolutePath()
	dstDir = dstFile.getAbsolutePath()
	failed = []
	for root, directories, filenames in os.walk(srcDir):
		filenames.sort()
		for index, filename in enumerate(filenames):
	  # Check for file extension
	  		if not filename.endswith(ext):
	  			continue
	  # Check for file name pattern
	  		if containString not in filename:
	  			continue
  			progress = round(float(index+1)/float(len(filenames)), 3)
  			print "Progress: %s of %s images (%s percent), Time: %s" %(index + 1, len(filenames), progress*100, formatTime(start))
  			
  			try:
				process(srcDir, dstDir, root, filename, keepDirectories)
			except:
				print("File %s failed"%filename)
				failed.append(filename)
	return failed

def process(srcDir, dstDir, currentDir, fileName, keepDirectories):
	print "Processing:"

  	# Opening the image
  	print "Open image file", fileName
  	imp = BF.openImagePlus(os.path.join(currentDir, fileName))[0]
  	title = imp.getTitle()
  	if imp.getNChannels() == 1:
  		imp = zStackIJ(imp)
  		imp.setTitle(title)
  		# Put your processing commands here!
  		imp, pred_imp, roi = ilastik_segmProbs(imp)
  # Saving the image
  		saveDir = currentDir.replace(srcDir, dstDir) if keepDirectories else dstDir
  		if not os.path.exists(saveDir):
  			os.makedirs(saveDir)
		print "Saving to", saveDir
		IJ.saveAs(imp, "Tiff", os.path.join(saveDir, fileName))
		
		pred_imp.setTitle("Segmented_"+title)
		pred_name = "Segmented_"+fileName
		IJ.saveAs(pred_imp, "Tiff", os.path.join(saveDir, pred_name))
		pred_imp.close()
		if not isinstance(roi, list):
			roi = [roi]
		print(roi)
		for index, r in enumerate(roi):
			r.setName("ROI_"+title)
			roi_filename = fileName.replace(ext, ".roi")
			roi_filename = "Roi%s_"%index + roi_filename
			imp.setRoi(r)
			
			IJ.saveAs(imp, "Selection", os.path.join(saveDir, roi_filename))
		imp.close()


def ilastik_segmentation(imp, pathToProject=ilastikPath.getAbsolutePath()):
	args = "projectfilename=%s inputimage=[%s] pixelclassificationtype=Segmentation"%(pathToProject, imp.getTitle())
	imp.show()
	print(args)
	IJ.run("Run Pixel Classification Prediction", args)
	imp.hide()
	pred_imp = IJ.getImage()
	pred_imp.hide()
	roi = getROI(pred_imp).getRoi()
	imp.setRoi(roi)
	return imp, pred_imp, roi

def ilastik_segmProbs(imp, pathToProject=ilastikPath.getAbsolutePath()):
	args = "projectfilename=%s inputimage=[%s] pixelclassificationtype=Segmentation"%(pathToProject, imp.getTitle())
	imp.show()
	IJ.run("Run Pixel Classification Prediction", args)
	imp.hide()
	pred_imp = IJ.getImage()
	#pred_imp.hide()
	
	rois = []
	for numC in range(1, numClass+1):
		print(numC)
		#roi = getROI(pred_imp).getRoi()
		IJ.run(pred_imp, "Manual Threshold...", "min=%s max=%s"%(numC, numC))
		IJ.run(pred_imp, "Create Selection", "")
		
		rois.append(pred_imp.getRoi())

	return imp, pred_imp, rois


def getROI(imp):
	IJ.setAutoThreshold(imp, "Default no-reset")
	IJ.run(imp, "Create Selection", "")
	return imp


failed = run()
print("Analysis Done!")
print("Failed Files: %s"%len(failed))
for f in failed:
	print(f)

#impPath = "Z:\ephacoffice\DColameo\DATA\Primary_Neurons_HT-APC\Test_Images_BF\MIN_9886_Pilot_CTX_GFP_tdTomato_Dapi_G3-1_2674.vsi - 077 BF.tif"
#imp = BF.openImagePlus(impPath)[0]
#imp.show()
#ilastikPath = "Z:\ephacoffice\DColameo\Ilastik\BF_Segmentation_one_Image.ilp"
#pred_imp = ilastik_segmentation(imp, ilastikPath)
#print(pred_imp)
#roi = getROI(pred_imp).getRoi()
#imp.setRoi(roi)
#imp.show()
#pred_imp.show()

