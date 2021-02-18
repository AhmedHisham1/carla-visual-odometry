# carla-visual-odometry

## What is this?
this is an implementation of visual odometry using stereo camera within CARLA simulator.

## How to test?
1. download the latest carla release from [their github repo releases page](https://github.com/carla-simulator/carla/releases). I am using the "CARLA_0.9.11.zip" windows release. If you choose the windows release, you just need to unzip the downloaded file.
2. Run the "WindowsNoEditor\CarlaUE4.exe" file to open the CARLA simulator server.
3. place the script "stereo_visual_odometry" in the following location "WindowsNoEditor\PythonAPI\examples", and run it.

# Results
#### [Youtube Video](https://youtu.be/pvq0h9L-e7Q)
#### Screenshots:
![result image](\images\res.jpg)


# Solution Steps
## 1. Calculating depth using left and right cameras' frames:
 1. Compute disparity of the left frame using opencv's `StereoSGBM` matcher.
 2. Calculate depth using the focal length, the baseline distance and the calculated disparity map. [depth = Z_c = f*b/d]

## 2. Extract and match features of each two consecutive frames from the left camera:
 1. Features extraction is done using `cv2.goodFeaturesToTrack` and ORB describtor.
 2. Matching is done using a brute force matcher. Matches are filtered according to a distance threshold to remove ambigous matches.

## 3. Motion Estimation:
![3D to 2D motion estimation](\images\3d-2d.jpg)

Using the calculated depth and the matches between the `t-1` and `t` frames the motion is estimated as follows:
1. Three values are inputed to the `cv2.solvePnPRansac()` solver:
    - objectpoints: 3D points in camera coordinates.
    - imagepoints: corresponding 2D points pixel values.
    - K: camera intrinsec parameters matrix.
    
    The solver returns the rotation and the translation vectors.
2. Get the rotation matrix `R` from the returned vector using `cv2.Rodrigues`

## 4. Update Trajectory
The previously calculated `[R|t]` matrix are used to calculate the new trajectory point as follow:

`RT = np.dot(RT, np.linalg.inv(rt_mtx))`

`new_trajectory = RT[:3, 3]`


---
# References
- "Visual Perception for Self-Driving Cars" course by university of torronto on Coursera:
    
    I couldn't have coded this without watching this course first. I even used some of my course-homework code in this.
- CARLA Simulator Documentation