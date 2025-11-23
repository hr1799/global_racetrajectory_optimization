#!/usr/bin/env python3

import os
import csv
import yaml
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize
from skimage.segmentation import watershed

from mincurv_trajectory_optimizer import trajectory_optimizer
import helper_funcs_glob
import trajectory_planning_helpers as tph


class GlobalPlanner():
    def __init__(self):
        self.pkg_dir = "/home/hariharan/roboracer/racetrajectory_optimization/"
        self.output_path = self.pkg_dir + "outputs/mincurv/"
        os.makedirs(self.output_path, exist_ok=True)
        self.safety_width = 0.35  # [m] including the width of the car
        self.map_name = "porto"
        self.map_img_ext = '.png'
        self.map_dir = "/home/hariharan/roboracer/roboracer_sim/maps"

        self.only_centerline = False
        self.show_plots = True

        # map variables
        self.map_resolution = 0.0
        self.map_origin = [0.0, 0.0]
        self.initial_position = [0.7, 0.0, -np.pi/2] # x, y, psi

        self.filter_length = 35

        with open(os.path.join(self.map_dir, self.map_name + '.yaml')) as f:
            data = yaml.safe_load(f)
            self.map_resolution = data['resolution']
            self.map_origin = data['origin']

        print('Global planner ready!')

    def compute_global_trajectory(self) -> bool:
        """
        Compute the global optimized trajectory of a map.

        Calculate the centerline of the track and compute global optimized trajectory with minimum curvature
        optimization.
        Publish the markers and waypoints of the global optimized trajectory.
        A waypoint has the following form: [s_m, x_m, y_m, d_right, d_left, psi_rad, vx_mps, ax_mps2]

        Returns
        -------
        bool
            True if successfully computed the global waypoints
        """
        ################################################################################################################
        # Create a filtered black and white image of the map
        ################################################################################################################
        img_path = os.path.join(self.map_dir, self.map_name + self.map_img_ext)
        image = cv2.imread(img_path)
        image = cv2.flip(image, 0)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        # Filtering with morphological opening
        kernel1 = np.ones((3, 3), np.uint8)
        img = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel1, iterations=2)

        # plt.imshow(img, cmap='gray', origin='lower')
        # plt.title('Original map after preprocessing')
        # plt.show()

        # get morphological skeleton of the map
        skeleton = skeletonize(img, method='lee')

        # f, (ax0, ax1) = plt.subplots(1,2)
        # ax0.imshow(img, cmap='gray', origin='lower')
        # ax0.set_title('img')
        # ax1.imshow(skeleton, cmap='gray', origin='lower')
        # ax1.set_title('Skeleton')
        # plt.show()

        ################################################################################################################
        # Extract centerline from filtered map
        ################################################################################################################
        try:
            centerline_meter_int, centerline_smooth = self.extract_centerline(skeleton=skeleton, show_plot=self.show_plots)
        except IOError:
            print('No closed contours found!')
            return

        ################################################################################################################
        # Extract track bounds
        ################################################################################################################
        try:
            bound_r_water, bound_l_water = self.extract_track_bounds(centerline_smooth, img)
            dist_transform = None
            print('Using watershed for track bound extraction...')
        except IOError:
            print('More than two track bounds detected with watershed algorithm')
            print('Trying with simple distance transform...')
            bound_r_water = None
            bound_l_water = None
            dist_transform = cv2.distanceTransform(img, cv2.DIST_L2, 5)
        
        ################################################################################################################
            
        el_lengths = np.linalg.norm(np.diff(centerline_meter_int, axis=0), axis=1)
        psi, kappa = tph.calc_head_curv_num.calc_head_curv_num(
            centerline_meter_int, el_lengths, False
        )
        centerline_nvecs = tph.calc_normal_vectors.calc_normal_vectors(psi)

        # plot centerline with normal vectors
        if self.show_plots:
            # Show the boundaries
            _, ax = plt.subplots()
            ax.plot(bound_r_water[:, 0], bound_r_water[:, 1], 'r', label='Right boundary')
            ax.plot(bound_l_water[:, 0], bound_l_water[:, 1], 'g', label='Left boundary')
            ax.plot(self.initial_position[0], self.initial_position[1], 'bo', label='Initial position')
            ax.legend()
            ax.plot(centerline_meter_int[:, 0], centerline_meter_int[:, 1], 'k', label='Centerline')
            ax.quiver(centerline_meter_int[:, 0], centerline_meter_int[:, 1], centerline_nvecs[:, 0], centerline_nvecs[:, 1], scale=2, scale_units='xy')
            plt.axis('equal')
            plt.show()

        cent_with_dist = self.add_dist_to_cent(centerline_smooth=centerline_smooth,
                                               centerline_meter=centerline_meter_int,
                                               dist_transform=dist_transform,
                                               bound_r=bound_r_water,
                                               bound_l=bound_l_water)
        
        centerline_bound_right = cent_with_dist[:, :2] + cent_with_dist[:, 2][:, None] * centerline_nvecs
        centerline_bound_left = cent_with_dist[:, :2] - cent_with_dist[:, 3][:, None] * centerline_nvecs

        # show the centerline with track bounds
        if self.show_plots:
            fig, ax = plt.subplots()
            ax.imshow(image, cmap='gray', origin='lower')
            ax.plot((cent_with_dist[:, 0] - self.map_origin[0]) / self.map_resolution, (cent_with_dist[:, 1] - self.map_origin[1]) / self.map_resolution, 'k', label='Centerline')
            ax.plot((centerline_bound_right[:, 0] - self.map_origin[0]) / self.map_resolution, (centerline_bound_right[:, 1] - self.map_origin[1]) / self.map_resolution, 'b', label='Right bound')
            ax.plot((centerline_bound_left[:, 0] - self.map_origin[0]) / self.map_resolution, (centerline_bound_left[:, 1] - self.map_origin[1]) / self.map_resolution, 'g', label='Left bound')
            plt.axis('equal')
            plt.legend()
            plt.show()

        # Write centerline in a csv file
        if self.only_centerline:
            centerline_fp = self.output_path + self.map_name + '_centerline.csv'
            output = np.column_stack((cent_with_dist, centerline_nvecs))
            np.savetxt(centerline_fp, output, delimiter=',')
        else:
            centerline_fp = self.output_path + self.map_name + '_centerline.csv'
            np.savetxt(centerline_fp, cent_with_dist, delimiter=',')

        ################################################################################################################
        # Compute global trajectory with mincurv_iqp optimization
        ################################################################################################################

        print('Start Global Trajectory optimization with iterative minimum curvature...')
        global_trajectory_iqp, bound_r_iqp, bound_l_iqp, est_t_iqp = trajectory_optimizer(pkg_dir=self.pkg_dir,
                                                                                          centerline_fp=centerline_fp,
                                                                                          safety_width=self.safety_width)
        # s_points_opt_interp,
        # raceline_interp_x,
        # raceline_interp_y,
        # psi_vel_opt,
        # kappa_opt,
        # vx_profile_opt,
        # ax_profile_opt))

        print('IQP estimated lap time: {} s'.format(round(est_t_iqp, 4)))
        print('IQP maximum speed: {} m/s'.format(round(np.amax(global_trajectory_iqp[:, 5]), 4)))

        raceline_meter_int = global_trajectory_iqp[:, 1:3]
        raceline_smooth = np.zeros(np.shape(raceline_meter_int))
        raceline_smooth[:, 0] = (raceline_meter_int[:, 0] - self.map_origin[0]) / self.map_resolution
        raceline_smooth[:, 1] = (raceline_meter_int[:, 1] - self.map_origin[1]) / self.map_resolution

        img_path = os.path.join(self.map_dir, self.map_name + self.map_img_ext)
        image = cv2.imread(img_path)
        image = cv2.flip(image, 0)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        # Filtering with morphological opening
        kernel1 = np.ones((3, 3), np.uint8)
        img = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel1, iterations=2)

        bound_r_water, bound_l_water = self.extract_track_bounds(raceline_smooth, img)

        nvecs = tph.calc_normal_vectors.calc_normal_vectors(global_trajectory_iqp[:, 3])

        # plot raceline with normal vectors
        if self.show_plots:
            fig, ax = plt.subplots()
            ax.plot(raceline_meter_int[:, 0], raceline_meter_int[:, 1], 'k', label='Centerline')
            ax.quiver(raceline_meter_int[:, 0], raceline_meter_int[:, 1], nvecs[:, 0], nvecs[:, 1], scale=2, scale_units='xy')
            plt.axis('equal')
            plt.show()

        raceline_with_dist = self.add_dist_to_cent(centerline_smooth=raceline_smooth,
                                               centerline_meter=raceline_meter_int,
                                               dist_transform=None,
                                               bound_r=bound_r_water,
                                               bound_l=bound_l_water)

        bound_right = raceline_with_dist[:, :2] + raceline_with_dist[:, 2][:, None] * nvecs
        bound_left = raceline_with_dist[:, :2] - raceline_with_dist[:, 3][:, None] * nvecs
        
        if self.show_plots:
            fig, ax = plt.subplots()
            ax.plot(raceline_with_dist[:, 0], raceline_with_dist[:, 1], 'k', label='Raceline')
            ax.plot(bound_left[:, 0], bound_left[:, 1], 'b', label='Right bound')
            ax.plot(bound_right[:, 0], bound_right[:, 1], 'g', label='Left bound')
            ax.plot(bound_l_water[:, 0], bound_l_water[:, 1], 'b--', label='Right bound watershed')
            ax.plot(bound_r_water[:, 0], bound_r_water[:, 1], 'g--', label='Left bound watershed')
            plt.axis('equal')
            plt.legend()
            plt.show()

        output = np.column_stack((global_trajectory_iqp, raceline_with_dist[:, 2:4], nvecs))
        np.savetxt(self.output_path + self.map_name + '_raceline.csv', output, delimiter=',')

        # plot the global trajectory
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray', origin='lower')
        ax.plot((cent_with_dist[:, 0] - self.map_origin[0]) / self.map_resolution, (cent_with_dist[:, 1] - self.map_origin[1]) / self.map_resolution, 'k--', label='Centerline')
        ax.plot((raceline_with_dist[:, 0] - self.map_origin[0]) / self.map_resolution, (raceline_with_dist[:, 1] - self.map_origin[1]) / self.map_resolution, 'k', label='Raceline')
        ax.plot((bound_right[:, 0] - self.map_origin[0]) / self.map_resolution, (bound_right[:, 1] - self.map_origin[1]) / self.map_resolution, 'b', label='Right bound')
        ax.plot((bound_left[:, 0] - self.map_origin[0]) / self.map_resolution, (bound_left[:, 1] - self.map_origin[1]) / self.map_resolution, 'g', label='Left bound')
        ax.plot((global_trajectory_iqp[:, 1] - self.map_origin[0]) / self.map_resolution, (global_trajectory_iqp[:, 2] - self.map_origin[1]) / self.map_resolution, 'r', label='Global trajectory')

        # safety bounds
        ax.plot((global_trajectory_iqp[:, 1] + self.safety_width/2 * nvecs[:, 0] - self.map_origin[0]) / self.map_resolution, (global_trajectory_iqp[:, 2] + self.safety_width/2 * nvecs[:, 1] - self.map_origin[1]) / self.map_resolution, 'g', label='Right safety bound')
        ax.plot((global_trajectory_iqp[:, 1] - self.safety_width/2 * nvecs[:, 0] - self.map_origin[0]) / self.map_resolution, (global_trajectory_iqp[:, 2] - self.safety_width/2 * nvecs[:, 1] - self.map_origin[1]) / self.map_resolution, 'b', label='Left safety bound')

        path = (global_trajectory_iqp[:, 1:3] -  np.array(self.map_origin[:2])) / self.map_resolution
        points = path.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(0, 8)
        lc = LineCollection(segments, cmap='jet', norm=norm)
        lc.set_array(global_trajectory_iqp[:, 5])
        lc.set_linewidth(5)
        line = plt.gca().add_collection(lc)
        plt.colorbar(line)
        plt.xlim(path[:, 0].min()-1, path[:, 0].max()+1)
        plt.ylim(path[:, 1].min()-1, path[:, 1].max()+1)
        plt.gca().set_aspect('equal', adjustable='box')

        plt.xticks([])
        plt.yticks([])
        plt.axis('equal')

        # save fig
        # plt.savefig(self.output_path + self.map_name + '_safety_width_{:.2f}'.format(self.safety_width) + '_global_trajectory.png', bbox_inches='tight', pad_inches=0.1)
        plt.legend()
        plt.show()
        
        return

    def extract_centerline(self, skeleton, show_plot = False) -> np.ndarray:
        """
        Extract the centerline out of the skeletonized binary image.

        This is done by finding closed contours and comparing these contours to the approximate centerline
        length (which is known because of the driven path).

        Parameters
        ----------
        skeleton
            The skeleton of the binarised and filtered map
        Returns
        -------
        centerline : np.ndarray
            The centerline in form [[x1,y1],...] and in cells not meters

        Raises
        ------
        IOError
            If no closed contour is found
        """
        # get contours from skeleton
        skeleton = (skeleton > 0).astype(np.uint8) * 255
        contours, hierarchy = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        print("Number of contours: ", len(contours))

        if len(contours) == 0:
            raise IOError("No contours found")
        elif len(contours) == 1:
            contour = contours[0]
        else:
            # show the contours
            for i, cont in enumerate(contours):
                plt.imshow(skeleton, cmap='gray', origin='lower')
                plt.plot(cont[:, 0, 0], cont[:, 0, 1], 'r', linewidth=2)
                plt.show()
            
            # ask the user to select the correct contour
            print("Please the index of the contour")
            raise IOError("Invalid index")
            contour = contours[index]

        # delete all points that are present more than once
        points = []
        for c in contour:
            x = c[0][0]
            y = c[0][1]
            if [x, y] not in points:
                points.append([x, y])
            else:
                points.remove([x, y])
        
        centerline = np.array(points)

        # show the centerline
        if self.show_plots:
            plt.imshow(skeleton, cmap='gray', origin='lower')
            plt.plot(centerline[:, 0], centerline[:, 1], 'r', linewidth=2)
            plt.show()

        centerline_smooth = self.smooth_centerline(centerline, self.filter_length)

        # show the centerline
        if self.show_plots:
            plt.imshow(skeleton, cmap='gray', origin='lower')
            plt.plot(centerline_smooth[:, 0], centerline_smooth[:, 1], 'r', linewidth=2)
            plt.show()

        # convert centerline from cells to meters
        centerline_meter = np.zeros(np.shape(centerline_smooth))
        centerline_meter[:, 0] = centerline_smooth[:, 0] * self.map_resolution + self.map_origin[0]
        centerline_meter[:, 1] = centerline_smooth[:, 1] * self.map_resolution + self.map_origin[1]

        # interpolate centerline to 0.1m stepsize: less computation needed later for distance to track bounds
        centerline_meter = np.column_stack((centerline_meter, np.zeros((centerline_meter.shape[0], 2))))

        centerline_meter_int = helper_funcs_glob.src.interp_track.interp_track(reftrack=centerline_meter,
                                                                               stepsize_approx=0.1)[:, :2]

        # get distance to initial position for every point on centerline
        distance_sq = np.power(centerline_meter_int[:, 0] - self.initial_position[0], 2) + \
                        np.power(centerline_meter_int[:, 1] - self.initial_position[1], 2)
        st_point_ind = np.argmin(distance_sq)
        
        # make sure the start point is the first point of the centerline
        centerline_meter_int = np.roll(centerline_meter_int, -st_point_ind, axis=0)
        centerline_smooth = np.roll(centerline_smooth, -st_point_ind, axis=0)
        
        st_point = centerline_meter_int[0]
        next_point = centerline_meter_int[1]
        centerline_vec = next_point - st_point
        centerline_vec = centerline_vec / np.linalg.norm(centerline_vec)
        cent_direction = np.arctan2(centerline_vec[1], centerline_vec[0])
        initial_vec = np.array([math.cos(self.initial_position[2]), math.sin(self.initial_position[2])])
        
        # flip centerline if directions don't match
        if np.dot(centerline_vec, initial_vec) < 0:
            print("Flipping centerline")
            centerline_smooth = np.flip(centerline_smooth, axis=0)
            centerline_meter_int = np.flip(centerline_meter_int, axis=0)
            cent_direction = cent_direction + math.pi

        if show_plot:
            print("Direction of the centerline: ", cent_direction)
            print("Direction of the initial car position: ", self.initial_position[2])
            plt.plot(centerline_meter_int[:, 0], centerline_meter_int[:, 1], 'ko', label='Centerline interpolated')
            plt.plot(centerline_meter_int[0, 0], centerline_meter_int[0, 1], 'ro', label='First point')
            plt.plot(centerline_meter_int[1, 0], centerline_meter_int[1, 1], 'bo', label='Second point')
            plt.legend()
            plt.axis('equal')
            plt.show()

        return centerline_meter_int, centerline_smooth

    @staticmethod
    def smooth_centerline(centerline: np.ndarray, filter_length: float) -> np.ndarray:
        """
        Smooth the centerline with a Savitzky-Golay filter.

        Notes
        -----
        The savgol filter doesn't ensure a smooth transition at the end and beginning of the centerline. That's why
        we apply a savgol filter to the centerline with start and end points on the other half of the track.
        Afterwards, we take the results of the second smoothed centerline for the beginning and end of the
        first centerline to get an overall smooth centerline

        Parameters
        ----------
        centerline : np.ndarray
            Unsmoothed centerline
        filter_length : float
            Length of the filter

        Returns
        -------
        centerline_smooth : np.ndarray
            Smooth centerline
        """

        print("Filter length: ", filter_length)
        centerline_smooth = savgol_filter(centerline, filter_length, 3, axis=0)

        # cen_len is half the length of the centerline
        cen_len = int(len(centerline) / 2)
        centerline2 = np.append(centerline[cen_len:], centerline[0:cen_len], axis=0)
        centerline_smooth2 = savgol_filter(centerline2, filter_length, 3, axis=0)

        # take points from second (smoothed) centerline for first centerline
        centerline_smooth[0:filter_length] = centerline_smooth2[cen_len:(cen_len + filter_length)]
        centerline_smooth[-filter_length:] = centerline_smooth2[(cen_len - filter_length):cen_len]

        return centerline_smooth

    def extract_track_bounds(self, centerline: np.ndarray, filtered_bw: np.ndarray) -> tuple:
        """
        Extract the boundaries of the track.

        Use the watershed algorithm with the centerline as marker to extract the boundaries of the filtered black
        and white image of the map.

        Parameters
        ----------
        centerline : np.ndarray
            The centerline of the track (in cells not meters)
        filtered_bw : np.ndarray
            Filtered black and white image of the track
        Returns
        -------
        bound_right, bound_left : tuple[np.ndarray, np.ndarray]
            Points of the track bounds right and left in meters

        Raises
        ------
        IOError
            If there were more (or less) than two track bounds found
        """
        # create a black and white image of the centerline
        cent_img = np.zeros((filtered_bw.shape[0], filtered_bw.shape[1]), dtype=np.uint8)
        cv2.drawContours(cent_img, [centerline.astype(int)], 0, 255, 2, cv2.LINE_8)

        # create markers for watershed algorithm
        _, cent_markers = cv2.connectedComponents(cent_img)

        # apply watershed algorithm to get only the track (without any lidar beams outside the track)
        dist_transform = cv2.distanceTransform(filtered_bw, cv2.DIST_L2, 5)
        labels = watershed(-dist_transform, cent_markers, mask=filtered_bw)

        closed_contours = []

        for label in np.unique(labels):
            if label == 0:
                continue

            # Create a mask, the mask should be the track
            mask = np.zeros(filtered_bw.shape, dtype="uint8")
            mask[labels == label] = 255

            if self.show_plots:
                plt.imshow(mask, cmap='gray', origin='lower')
                plt.title('Mask')
                plt.show()

            # Find contours
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

            # save all closed contours
            for i, cont in enumerate(contours):
                opened = hierarchy[0][i][2] < 0 and hierarchy[0][i][3] < 0
                if not opened:
                    closed_contours.append(cont)

            # there must not be more (or less) than two closed contour
            if len(closed_contours) != 2:
                raise IOError(f"More than two track bounds ({len(closed_contours)}) detected! Check input")
            # draw the boundary into the centerline image
            cv2.drawContours(cent_img, closed_contours, 0, 255, 4)
            cv2.drawContours(cent_img, closed_contours, 1, 255, 4)

        # the longest closed contour is the outer boundary
        bound_long = max(closed_contours, key=len)
        bound_long = np.array(bound_long).flatten()

        # inner boundary is the shorter one
        bound_short = min(closed_contours, key=len)
        bound_short = np.array(bound_short).flatten()

        # reshape from the shape [x1,y1,x2,y2,...] to [[x1,y1],[x2,y2],...]
        len_reshape = int(len(bound_long) / 2)
        bound_long = bound_long.reshape(len_reshape, 2)
        # convert to meter
        bound_long_meter = np.zeros(np.shape(bound_long))
        bound_long_meter[:, 0] = bound_long[:, 0] * self.map_resolution + self.map_origin[0]
        bound_long_meter[:, 1] = bound_long[:, 1] * self.map_resolution + self.map_origin[1]

        # reshape from the shape [x1,y1,x2,y2,...] to [[x1,y1],[x2,y2],...]
        len_reshape = int(len(bound_short) / 2)
        bound_short = bound_short.reshape(len_reshape, 2)
        # convert to meter
        bound_short_meter = np.zeros(np.shape(bound_short))
        bound_short_meter[:, 0] = bound_short[:, 0] * self.map_resolution + self.map_origin[0]
        bound_short_meter[:, 1] = bound_short[:, 1] * self.map_resolution + self.map_origin[1]

        # get distance to initial position for every point on the outer bound to figure out if it is the right
        # or left boundary
        bound_distance = np.sqrt(np.power(bound_long_meter[:, 0] - self.initial_position[0], 2)
                                 + np.power(bound_long_meter[:, 1] - self.initial_position[1], 2))

        min_dist_ind = np.argmin(bound_distance)

        bound_direction = np.angle([complex(bound_long_meter[min_dist_ind, 0] - bound_long_meter[min_dist_ind - 1, 0],
                                            bound_long_meter[min_dist_ind, 1] - bound_long_meter[min_dist_ind - 1, 1])])

        norm_angle_right = self.initial_position[2] - math.pi
        if norm_angle_right < -math.pi:
            norm_angle_right = norm_angle_right + 2 * math.pi

        if self.same_direction(norm_angle_right, bound_direction):
            bound_right = bound_long_meter
            bound_left = bound_short_meter
        else:
            bound_right = bound_short_meter
            bound_left = bound_long_meter

        return bound_right, bound_left

    def dist_to_bounds(self, trajectory: np.ndarray, bound_r, bound_l, centerline: np.ndarray):
        """
        Calculate the distance to track bounds for every point on a trajectory.

        Parameters
        ----------
        trajectory : np.ndarray
            A trajectory in form [s_m, x_m, y_m, psi_rad, vx_mps, ax_mps2] or [x_m, y_m]
        bound_r
            Points in meters of boundary right
        bound_l
            Points in meters of boundary left
        centerline : np.ndarray
            Centerline only needed if global trajectory is given and plot of it is wanted

        Returns
        -------
        dists_right, dists_left : tuple[np.ndarray, np.ndarray]
            Distances to the right and left track boundaries for every waypoint
        """
        # check format of trajectory
        if len(trajectory[0]) > 2:
            help_trajectory = trajectory[:, 1:3]
        else:
            help_trajectory = trajectory

        # interpolate track bounds
        bound_r_tmp = np.column_stack((bound_r, np.zeros((bound_r.shape[0], 2))))
        bound_l_tmp = np.column_stack((bound_l, np.zeros((bound_l.shape[0], 2))))

        bound_r_int = helper_funcs_glob.src.interp_track.interp_track(reftrack=bound_r_tmp,
                                                                      stepsize_approx=0.1)
        bound_l_int = helper_funcs_glob.src.interp_track.interp_track(reftrack=bound_l_tmp,
                                                                      stepsize_approx=0.1)

        # find the closest points of the track bounds to global trajectory waypoints
        n_wpnt = len(help_trajectory)
        dists_right = np.zeros(n_wpnt)  # contains (min) distances between waypoints and right bound
        dists_left = np.zeros(n_wpnt)  # contains (min) distances between waypoints and left bound

        for i, wpnt in enumerate(help_trajectory):
            dists_bound_right = np.sqrt(np.power(bound_r_int[:, 0] - wpnt[0], 2)
                                        + np.power(bound_r_int[:, 1] - wpnt[1], 2))
            dists_right[i] = np.amin(dists_bound_right)

            dists_bound_left = np.sqrt(np.power(bound_l_int[:, 0] - wpnt[0], 2)
                                       + np.power(bound_l_int[:, 1] - wpnt[1], 2))
            dists_left[i] = np.amin(dists_bound_left)

        return dists_right, dists_left

    def add_dist_to_cent(self, centerline_smooth: np.ndarray,
                         centerline_meter: np.ndarray, dist_transform=None,
                         bound_r: np.ndarray = None, bound_l: np.ndarray = None) -> np.ndarray:
        """
        Add distance to track bounds to the centerline points.

        Parameters
        ----------
        centerline_smooth : np.ndarray
            Smooth centerline in cells (not meters)
        centerline_meter : np.ndarray
            Smooth centerline in meters (not cells)
        dist_transform : Any, default=None
            Euclidean distance transform of the filtered black and white image
        bound_r : np.ndarray, default=None
            Points of the right track bound in meters
        bound_l : np.ndarray, default=None
            Points of the left track bound in meters

        Returns
        -------
        centerline_comp : np.ndarray
            Complete centerline with distance to right and left track bounds for every point
        """
        centerline_comp = np.zeros((len(centerline_meter), 4))

        if dist_transform is not None:
            width_track_right = dist_transform[centerline_smooth[:, 1].astype(int),
                                               centerline_smooth[:, 0].astype(int)] * self.map_resolution
            if len(width_track_right) != len(centerline_meter):
                width_track_right = np.interp(np.arange(0, len(centerline_meter)), np.arange(0, len(width_track_right)),
                                              width_track_right)
            width_track_left = width_track_right
        elif bound_r is not None and bound_l is not None:
            width_track_right, width_track_left = self.dist_to_bounds(centerline_meter, bound_r, bound_l,
                                                                      centerline=centerline_meter)
        else:
            raise IOError("No closed contours found...")

        centerline_comp[:, 0] = centerline_meter[:, 0]
        centerline_comp[:, 1] = centerline_meter[:, 1]
        centerline_comp[:, 2] = width_track_right 
        centerline_comp[:, 3] = width_track_left 
        return centerline_comp

    @staticmethod
    def same_direction(alpha: float, beta: float) -> bool:
        """
        Compare the direction of two points and check if they point in the same direction.

        Parameters
        ----------
        alpha : float
            direction angle in rad
        beta : float
            direction angle in rad

        Returns
        -------
        bool
            True if alpha and beta point in the same direction
        """
        delta_theta = math.fabs(alpha - beta)

        if delta_theta > math.pi:
            delta_theta = 2 * math.pi - delta_theta

        return delta_theta < math.pi / 2


if __name__ == "__main__":
    planner = GlobalPlanner()
    planner.compute_global_trajectory()
