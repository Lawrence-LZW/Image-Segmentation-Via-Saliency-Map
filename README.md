# Image-Segmentation-Via-Saliency-Map
The aim of this project is to perform image segmentation on the saliency map built from the image input.<br/>

Four image were renamed as tester.These tester can be used to test the algorithm<br/>

# Platform and Libraries
Python 3 is used as the platform to build saliency map.Python 3.9 interpreter was used.<br/>
User is required to install packages mentioned <br/>
Libraries and recommend version is listed down<br/>
1.OpenCV(4.5.5.64)<br/>
2.numpy(1.22.0)<br/>
3.Python Imaging Library(PIL)(1.5.33)<br/>
4.matplotlib(3.5.1)<br/>

Matlab is used as the platform to perform image segmentation<br/>
User is required to install Image Segmenter from Image Processing and Computer Vision app.<br/>
This can be done by install  Image Processing and Computer Vision in the Apps tab.<br/>

# How to Run
## Saliency Map
1.Ensure all the required libraries is available<br/>
2.Open the file pyall.py in any Python IDE<br/>
3.Run the code<br/>
4.Saliency Map is produced and shown<br/>
## Image Segmentation on Saliency Map
1.Find the saliency map produced in the same path with the pyall.py file<br/>
2.Open Matlab and open Image Segmenter from the Apps tab<br/>
3.Load saliency map by using Load Image function<br/>
4.Select ROI(Region of Interest) in ADD TO MASK<br/>
5.Draw an area that cover the object and apply the ROI<br/>
6.Click Active Contours in the REFINE MASK<br/>
7.Set a number of iterations. Suggestion：set the number as a higher value such as 900.The number can be scale down depends on the result<br/>
8.Click the Run button<br/>
9.The object will be segment out from the saliency map input<br/>
