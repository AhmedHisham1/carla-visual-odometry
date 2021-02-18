import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import numpy as np
import cv2
import time
from queue import Queue
from queue import Empty

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class VehicleEnv():
    def __init__(self, vehicle_name='model3', fps=20):
        self.actors = []
        self.delta_seconds = 1/fps
        self.sensor_queue = Queue()
        self.left_frame = None
        self.right_frame = None

        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)

        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

        self.vehicle_bp = np.random.choice(self.blueprint_library.filter(vehicle_name))
        self.start_pose = np.random.choice(self.world.get_map().get_spawn_points())
        self.waypoint = self.world.get_map().get_waypoint(self.start_pose.location)
        self.vehicle = self.world.spawn_actor(self.vehicle_bp, self.start_pose)
        self.vehicle.set_simulate_physics(False)
        self.actors.append(self.vehicle)

        self.camera_bp = self.blueprint_library.find('sensor.camera.rgb')

        self.left_camera_t = carla.Transform(carla.Location(x=2, y=-0.2, z=1.4))
        self.right_camera_t = carla.Transform(carla.Location(x=2, y=0.2, z=1.4))
        
        self.left_camera = self.world.spawn_actor(self.camera_bp, 
                                                  self.left_camera_t, 
                                                  attach_to=self.vehicle)
        self.actors.append(self.left_camera)
        self.right_camera = self.world.spawn_actor(self.camera_bp, 
                                                  self.right_camera_t, 
                                                  attach_to=self.vehicle)
        self.actors.append(self.right_camera)


        # Build the K projection matrix:
        # K = [[Fx,  0, image_w/2],
        #      [ 0, Fy, image_h/2],
        #      [ 0,  0,         1]]
        self.image_w = self.camera_bp.get_attribute("image_size_x").as_int()
        self.image_h = self.camera_bp.get_attribute("image_size_y").as_int()
        self.fov = self.camera_bp.get_attribute("fov").as_float()
        self.focal = self.image_w / (2.0 * np.tan(self.fov * np.pi / 360.0))

        # In this case Fx and Fy are the same since the pixel aspect ratio is 1
        self.K = np.identity(3)
        self.K[0, 0] = self.K[1, 1] = self.focal
        self.K[0, 2] = self.image_w / 2.0
        self.K[1, 2] = self.image_h / 2.0
        print(f'K matrix: {self.K}')

        self.left_camera.listen(lambda data: self._capture_frame(data, "left"))
        self.right_camera.listen(lambda data: self._capture_frame(data, "right"))

        self.original_settings = self.world.get_settings()
        self.world.apply_settings(
            carla.WorldSettings(no_rendering_mode=False, synchronous_mode=True, 
                                fixed_delta_seconds=self.delta_seconds))

        self.world_frame = self.world.get_snapshot().frame


    def visualize_trajectory(self):
        try:
            # initially
            world2camera = np.array(self.left_camera.get_transform().get_inverse_matrix())
            actual_trajectory = self.left_camera.get_location()
            # actual_trajectory = self.waypoint.transform.location
            actual_trajectory = np.array([actual_trajectory.x, actual_trajectory.y, actual_trajectory.z]).reshape((3,1))

            trajectory = np.array([0,0,0]).reshape((3,1))
            R = np.diag([1,1,1])
            T = trajectory
            RT = np.hstack([R, T])
            RT = np.vstack([RT, np.array([0,0,0,1])])

            old_frame = None
            
            self.fig = plt.figure(figsize=(15,4))
            self.ax = self.fig.add_subplot(1,4,1, projection='3d')
            self.ax2 = self.fig.add_subplot(1,4,2)
            self.ax2.set_aspect('equal', adjustable='box')
            self.ax3 = self.fig.add_subplot(1,4,3)
            self.ax3.set_aspect('equal', adjustable='box')
            self.ax4 = self.fig.add_subplot(1,4,4)
            self.ax4.set_aspect('equal', adjustable='box')
            while True:
                self._update_world()
                actual_trajectory_latest = self.left_camera.get_location()
                # actual_trajectory_latest = self.waypoint.transform.location
                actual_trajectory_latest = np.array([actual_trajectory_latest.x, actual_trajectory_latest.y, actual_trajectory_latest.z]).reshape((3,1))
                actual_trajectory = np.hstack([actual_trajectory, actual_trajectory_latest])
                
                if (self.left_frame is not None) and (self.right_frame is not None):
                    lf = self.left_frame
                    rf = self.right_frame
                    cur_frame = lf.copy()
                    self._show_frames(lf, rf)

                    depth = self._calculate_depth(lf, rf)
                
                if old_frame is not None:
                    # features detection and matching
                    kp1, des1 = self._extract_features(old_frame)
                    kp2, des2 = self._extract_features(cur_frame)
                    match = self._match_features(des1, des2)
                    match = self._filter_matches_distance(match, dist_threshold=0.6)

                    # updating trajectory
                    trajectory, RT = self._update_trajectory(trajectory, RT, self._estimate_motion, 
                                                             match, kp1, kp2, self.K, depth=depth)

                    # mapping actual trajectory from world axes to the camera axes [refer to lidar_to_camera.py example]
                    world_points = np.vstack([actual_trajectory, np.ones((1, actual_trajectory.shape[1]))])
                    cam_points = np.dot(world2camera, world_points)
                    cam_points = np.array([cam_points[1],
                                           cam_points[2]*-1,
                                           cam_points[0]])
                    
                    # draw trajectory
                    print(f'trajectory: {trajectory.shape}')
                    print(f'cam_points: {cam_points.shape}')
                    self._draw_trajectory(estimated=trajectory, actual=cam_points)

                    depth = ((depth - depth.mean()) / depth.std()) * 255
                    depth = depth.astype(np.uint8)
                    ret = self._show_frame(depth, name='depth')
                    # ret2 = self._show_frame(old_frame, name='oldframe')
                    # ret3 = self._show_frame(cur_frame, name='curframe')
                    if (ret == 0):
                        print('*'*20, 'ret==0!', '*'*20, sep='\n')
                        break
                
                old_frame = cur_frame.copy()
        except Exception as e:
            print('*'*15, e, type(e).__name__, e.__traceback__.tb_lineno, '*'*15, sep='\n')
        finally:
            self.world.apply_settings(self.original_settings)
            print('*'*15, 'Finally! Bye!', '*'*15, sep='\n')
            for actor in self.actors:
                actor.destroy()
                print(f'actor "{actor}" was destroyed!')


    def _capture_frame(self, sensor_data, sensor_name):
        frame = np.array(sensor_data.raw_data, dtype=np.uint8)
        frame = frame.reshape((sensor_data.height, sensor_data.width, 4))[...,:3]
        self.sensor_queue.put((frame, sensor_name))

    def _update_world(self):
        self.world.tick()
        self.world_frame = self.world.get_snapshot().frame
        print(f'world frame: {self.world_frame}')
        self.waypoint = np.random.choice(self.waypoint.next(1))
        self.vehicle.set_transform(self.waypoint.transform)

        for _ in range(2):
            s_frame = self.sensor_queue.get(True, 1.0)
            if s_frame[1] == 'right':
                self.right_frame = s_frame[0]
            elif s_frame[1] == 'left':
                self.left_frame = s_frame[0]

    def _show_frame(self, img, name='frame'):
        cv2.imshow(name, img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return 0
    
    def _show_frames(self, lf, rf):
        cv2.imshow('left', lf)
        cv2.imshow('right', rf)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return 0

    def _compute_left_disparity(self, img_left, img_right):
        num_disparities = 6*16
        block_size = 11
        min_disparity = 0
        window_size = 6
        
        img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        
        # Stereo BM matcher
        # left_matcher_BM = cv2.StereoBM_create(numDisparities=num_disparities,
        #                                       blockSize=block_size)

        # disp_left = left_matcher_BM.compute(img_left, img_right).astype(np.float32)/16

        # Stereo SGBM matcher
        left_matcher_SGBM = cv2.StereoSGBM_create(minDisparity=min_disparity,
                                                  numDisparities=num_disparities,
                                                  blockSize=block_size,
                                                  P1=8 * 3 * window_size ** 2,
                                                  P2=32 * 3 * window_size ** 2,
                                                  mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

        disp_left = left_matcher_SGBM.compute(img_left, img_right).astype(np.float32)/16
        
        return disp_left

    def _calculate_depth(self, lf, rf):
        disp_left = self._compute_left_disparity(lf, rf)
        f = self.K[0,0]
        b = 0.4
        # Replace all instances of 0 and -1 disparity with a small minimum value (to avoid div by 0 or negatives)
        disp_left[disp_left == 0] = 0.1
        disp_left[disp_left == -1] = 0.1
        depth_map = np.ones(disp_left.shape, np.single)
        depth_map[:] = f * b / disp_left[:]

        return depth_map

    def _extract_features(self, image):
        # orb = cv2.ORB_create(nfeatures=1000)
        # kp, des = orb.detectAndCompute(image,None)

        orb = cv2.ORB_create()
        kp = cv2.goodFeaturesToTrack(np.mean(image, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=7)
        kp = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in kp]
        kp, des = orb.compute(image, kp)
        
        return kp, des

    def _match_features(self, des1, des2):
    #     bfmatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    #     match = bfmatcher.match(des1,des2)
        bfmatcher = cv2.BFMatcher()
        match = bfmatcher.knnMatch(des1,des2, k=2)
        return match

    def _filter_matches_distance(self, match, dist_threshold):
        filtered_match = []
        for i, (m,n) in enumerate(match):
            if m.distance <= dist_threshold*n.distance:
                filtered_match.append(m)

        return filtered_match

    def _estimate_motion(self, match, kp1, kp2, k, depth=None):
        image1_points = []
        image2_points = []
        objectpoints = []
        
        if depth is not None:
            for m in match:
                u1, v1 = kp1[m.queryIdx].pt
                u2, v2 = kp2[m.trainIdx].pt
                s = depth[int(v1), int(u1)]
                if s < 1e3:
                    p_c = np.linalg.inv(k) @ (s * np.array([u1, v1, 1]))
                    image1_points.append([u1, v1])
                    image2_points.append([u2, v2])
                    objectpoints.append(p_c)

            try:
                objectpoints = np.vstack(objectpoints)
                imagepoints = np.array(image2_points)
                _, rvec, tvec, _ = cv2.solvePnPRansac(objectpoints, imagepoints, k, None)
                # _, rvec, tvec = cv2.solvePnP(objectpoints, imagepoints, k, None)
                rmat, _ = cv2.Rodrigues(rvec)
            except Exception as e:
                print('*'*15, e, type(e).__name__, e.__traceback__.tb_lineno, '*'*15, sep='\n')

        else:
            for m in match:
                train_idx = m.trainIdx
                query_idx = m.queryIdx
                
                p1x, p1y = kp1[query_idx].pt
                image1_points.append([p1x, p1y])

                p2x, p2y = kp2[train_idx].pt
                image2_points.append([p2x, p2y])

            E, _ = cv2.findEssentialMat(np.array(image1_points), np.array(image2_points), k, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            
            try:
                retval, rmat, tvec, _ = cv2.recoverPose(E, np.array(image1_points), np.array(image2_points), k)
            #     rmat, _, tvec = cv2.decomposeEssentialMat(E)
            except:
                print(f'E shape: {E.shape}')
                print(f'image1_points shape: {np.array(image1_points).shape}')
                print(f'image2_points shape: {np.array(image2_points).shape}')
                raise 'Cannot recover pose from this E matrix'
        
        return rmat, tvec

    def _update_trajectory(self, prev_traj, prev_RT, estimate_motion, match, kp1, kp2, k, depth=None):
        trajectory = [prev_traj[:, i] for i in range(prev_traj.shape[-1])]

        RT = prev_RT
        # print(RT)
        rmat, tvec = estimate_motion(match, kp1, kp2, k, depth=depth)
        rt_mtx = np.hstack([rmat, tvec])
        rt_mtx = np.vstack([rt_mtx, np.array([0,0,0,1])])
        RT = np.dot(RT, np.linalg.inv(rt_mtx))
        # print(RT)
        new_trajectory = RT[:3, 3]
        trajectory.append(new_trajectory)
        
        trajectory = np.array(trajectory).T
        
        return trajectory, RT

    def _draw_trajectory(self, estimated, actual):
        plt.cla()
        self.ax.clear()
        self.ax.plot3D(estimated[0, :], estimated[1, :], estimated[2, :], 'r', label='estimated')
        self.ax.scatter3D(estimated[0, :], estimated[1, :], estimated[2, :], c='r')
        self.ax.plot3D(actual[0, :], actual[1, :], actual[2, :], 'b', label='actual')
        self.ax.scatter3D(actual[0, :], actual[1, :], actual[2, :], c='b')
        self.ax.set_title('3D')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.legend()

        self.ax2.clear()
        self.ax2.plot(estimated[0, :], estimated[1, :], 'r', label='estimated')
        self.ax2.scatter(estimated[0, :], estimated[1, :], c='r')
        self.ax2.plot(actual[0, :], actual[1, :], 'b', label='actual')
        self.ax2.scatter(actual[0, :], actual[1, :], c='b')
        self.ax2.set_xlabel('X')
        self.ax2.set_ylabel('Y')
        self.ax2.set_title('x-y')

        self.ax3.clear()
        self.ax3.plot(estimated[0, :], estimated[2, :], 'r', label='estimated')
        self.ax3.scatter(estimated[0, :], estimated[2, :], c='r')
        self.ax3.plot(actual[0, :], actual[2, :], 'b', label='actual')
        self.ax3.scatter(actual[0, :], actual[2, :], c='b')
        self.ax3.set_xlabel('X')
        self.ax3.set_ylabel('Z')
        self.ax3.set_title('x-z')

        self.ax4.clear()
        self.ax4.plot(estimated[1, :], estimated[2, :], 'r', label='estimated')
        self.ax4.scatter(estimated[1, :], estimated[2, :], c='r')
        self.ax4.plot(actual[1, :], actual[2, :], 'b', label='actual')
        self.ax4.scatter(actual[1, :], actual[2, :], c='b')
        self.ax4.set_title('y-z')
        self.ax4.set_xlabel('Y')
        self.ax4.set_ylabel('Z')

        self.fig.canvas.draw()
        fig_img = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        fig_img = fig_img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        self._show_frame(cv2.cvtColor(fig_img, cv2.COLOR_RGB2BGR), name='figure')

if __name__ == "__main__":
    myEnv = VehicleEnv()
    myEnv.visualize_trajectory()
