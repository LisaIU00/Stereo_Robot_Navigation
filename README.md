# Stereo_Robot_Navigation
Project of the course "Computer Vision and Image Processing M" at the university of Bologna.

Alma Mater Studiorum UNIBO - Master of Science in Computer Engineering

Professor Di Stefano L.

# How to run
* clone repository
* run file 'main.py'
# Objective 
Given a video sequence taken by a stereo camera mounted on a moving vehicle, project’s objective is to sense information concerning the space in front of the vehicle which may be deployed by the vehicle navigation system to automatically avoid obstacles.
# Dataset
The input data consist of a pair of synchronized videos taken by a stereo camera (*robotL.avi*, *robotR.avi*), with one video concerning the left view (*robotL.avi*), the other the right view (*robotR.avi*). Moreover, the parameters required to estimate distances from stereo images are provided below:
* focale f = 567.2 pixel
* baseline b = 92.226 mm
# Functional Specifications
Sensing of 3D information related to the obstacles in front of the vehicle should rely on the stereo vision principle. Purposely, students should develop an area-based stereo matching algorithm capable of producing a dense disparity map for each pair of synchronized frames and based on the SAD (Sum of Absolute Differences)
dissimilarity measure.

For each pair of candidate corresponding points, the basic stereo matching algorithm consists in comparing the intensities belonging to two squared windows centred at the points. Such a comparison involves computation of either a dissimilarity (e.g. SAD, SSD) or similarity (e.g. NCC, ZNCC) measure between the two windows. As the matching process is carried out on rectified images, once a reference image is chosen (e.g. the left view), the candidates associated with a given point need to be sought for along the same row in the other image (right view) only and, usually, within a certain disparity range which depends on the depth range one wishes to sense. Accordingly, given a point in the reference image, the corresponding one in the other image is selected as the candidate minimizing (maximizing) the chosen dissimilarity (similarity) measure between the windows. As such, the parameters of the basic stereo matching algorithm consist in the size of the window and the disparity range. In this project, students should choose the former properly, while the latter is fixed to the interval [0,128].

The main task of the project requires the following steps:
1. Computing the disparity map in a central area of the reference frame (e.g. a squared area of size 60x60, 80x80 o 100x100 pixels), so to sense distances in the portion of the environment which would be travelled by the vehicle should it keep a straight trajectory.
2. Estimate a *main disparity* (**dmain**) for the frontal (wrt the camera) portion of the environment based on the disparity map of the central area of the reference frame computed in the previous step, e.g. by choosing the average disparity or the most frequent disparity within the map.
3. Determine the distance (z, in *mm*) of the obstacle wrt to the moving vehicle based on the *main disparities* (in pixel) estimated from each pair of frames:

   $$ z(mm) = {b(mm)*f(pixel) \over dmain(pixel)} $$

4. Generate a suitable output to convey to the user, in each pair of frame, the information related to the distance (converted in meters) from the camera to the obstacle. Moreover, an alarm should be generated whenever the distance turns out below 0.8 meters
5. Compute the real dimensions in *mm* (**W,H**) of the chessboard pattern present in the scene.
   Purposely, the OpenCV functions cvFindChessboardCorners and cvDrawChessboardCorners may be deployed to, respectively, find and display the pixel coordinates of the internal corners of the chessboard. Then, assuming the chessboard pattern to be parallel to the image plane of the stereo sensor, the real dimensions of the pattern can be obtained from their pixel dimensions (w,h) by the following formulas:

  $$ W(mm) = {z(mm)*w(pixel) \over f(pixel)} $$

  $$ H(mm) = {z(mm)*h(pixel) \over f(pixel)} $$

  Moreover, students should compare the estimated real dimensions to the known ones (125 *mm* x 178 *mm*) during the first approach manoeuvre of the vehicle to the pattern, so to verify that accuracy becomes higher as the vehicle gets closer to the pattern. Students should also comment on why accuracy turns out worse during the second approach manoeuvre.


