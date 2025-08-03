import numpy as np
import time
import json
import os
from typing import Tuple
import trajectory_planning_helpers as tph
import matplotlib.pyplot as plt
import configparser
import helper_funcs_glob

"""
Created by:
Luca Schwarzenbach

Documentation:
This is an adaption of the main_globaltraj.py file which allows to call a function to optimize a trajectory and returns
the optimized trajectory, the track boundaries, and the lap time.
"""        
        
def trajectory_optimizer(pkg_dir: str,
                         centerline_fp: str,
                         safety_width: float = 0.8,
                         plot: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Optimizes a trajectory based on the given input parameters.

    Parameters:
    - pkg_dir (str): The package directory where the inputs, params, and outputs are located.
    - centerline_fp (str): The file path to the centerline of the track.
    - safety_width (float): The safety width for the trajectory. Default is 0.8.
    - plot (bool): Whether to plot the optimized trajectory. Default is False.

    Returns:
    - Tuple[np.ndarray, np.ndarray, np.ndarray, float]: A tuple containing the optimized trajectory, the track boundaries, and the lap time.

    Raises:
    - IOError: If the optimization type is unknown.
    - ValueError: If the vehicle parameter file does not exist or is empty.

    """
    # debug and plot options -------------------------------------------------------------------------------------------
    debug = True                                    # print console messages
    plot_opts = {"mincurv_curv_lin": False,         # plot curv. linearization (original and solution based, mincurv only)
                 "raceline": True,                  # plot optimized path
                 "imported_bounds": False,          # plot imported bounds (analyze difference to interpolated bounds)
                 "raceline_curv": True,             # plot curvature profile of optimized path
                 "racetraj_vel": True,              # plot velocity profile
                 "racetraj_vel_3d": True,          # plot 3D velocity profile above raceline
                 "racetraj_vel_3d_stepsize": 0.2,   # [m] vertical lines stepsize in 3D velocity profile plot
                 "spline_normals": False,           # plot spline normals to check for crossings
                 "mintime_plots": False}            # plot states, controls, friction coeffs etc. (mintime only)
    
    # vehicle parameter file -------------------------------------------------------------------------------------------
    file_paths = {"veh_params_file": "racecar_f110.ini"}

    # set import options -----------------------------------------------------------------------------------------------
    imp_opts = {"flip_imp_track": False,                # flip imported track to reverse direction
                "set_new_start": False,                 # set new starting point (changes order, not coordinates)
                "new_start": np.array([-2.4, 0.0]),    # [x_m, y_m]
                "min_track_width": None,                # [m] minimum enforced track width (set None to deactivate)
                "num_laps": 1}                          # number of laps to be driven (significant with powertrain-option),
                                                        # only relevant in mintime-optimization

    # ------------------------------------------------------------------------------------------------------------------
    # INITIALIZATION OF PATHS ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    
    # input file path --------------------------------------------------------------------------------------------------
    file_paths["inputs"] = os.path.join(pkg_dir, "inputs")
    
    # get current path
    file_paths["module"] = os.path.dirname(os.path.abspath(__file__))

    # assemble track import path
    file_paths["track_file"] = centerline_fp

    # assemble export paths
    file_paths["traj_race_export"] = os.path.join(file_paths["module"], "outputs", "traj_race_cl.csv")
    file_paths["mintime_export"] = os.path.join(file_paths["module"], "outputs", "mintime")

    # ------------------------------------------------------------------------------------------------------------------
    # IMPORT VEHICLE DEPENDENT PARAMETERS ------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # load vehicle parameter file into a "pars" dict
    parser = configparser.ConfigParser()
    pars = {}

    if not parser.read(os.path.join(pkg_dir, "params", file_paths["veh_params_file"])):
        raise ValueError('Specified config file does not exist or is empty!, Looking for: {}'.format(
            os.path.join(pkg_dir, "params", file_paths["veh_params_file"])))

    pars["ggv_file"] = json.loads(parser.get('GENERAL_OPTIONS', 'ggv_file'))
    pars["ax_max_machines_file"] = json.loads(parser.get('GENERAL_OPTIONS', 'ax_max_machines_file'))
    pars["stepsize_opts"] = json.loads(parser.get('GENERAL_OPTIONS', 'stepsize_opts'))
    pars["reg_smooth_opts"] = json.loads(parser.get('GENERAL_OPTIONS', 'reg_smooth_opts'))
    pars["veh_params"] = json.loads(parser.get('GENERAL_OPTIONS', 'veh_params'))
    pars["vel_calc_opts"] = json.loads(parser.get('GENERAL_OPTIONS', 'vel_calc_opts'))
    pars["optim_opts"] = json.loads(parser.get('OPTIMIZATION_OPTIONS', 'optim_opts_mincurv'))

    # set import path for ggv diagram and ax_max_machines
    file_paths["ggv_file"] = os.path.join(file_paths["inputs"], "veh_dyn_info", pars["ggv_file"])
    file_paths["ax_max_machines_file"] = os.path.join(file_paths["inputs"], "veh_dyn_info",
                                                        pars["ax_max_machines_file"])

    # ------------------------------------------------------------------------------------------------------------------
    # IMPORT TRACK AND VEHICLE DYNAMICS INFORMATION --------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # save start time
    t_start = time.perf_counter()

    # import track
    reftrack_imp = helper_funcs_glob.src.import_track.import_track(imp_opts=imp_opts,
                                                                   file_path=file_paths["track_file"],
                                                                   width_veh=pars["veh_params"]["width"])

    # import ggv and ax_max_machines
    ggv, ax_max_machines = tph.import_veh_dyn_info.\
        import_veh_dyn_info(ggv_import_path=file_paths["ggv_file"],
                            ax_max_machines_import_path=file_paths["ax_max_machines_file"])

    # ------------------------------------------------------------------------------------------------------------------
    # PREPARE REFTRACK -------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    reftrack_interp, normvec_normalized_interp, a_interp, coeffs_x_interp, coeffs_y_interp = \
        helper_funcs_glob.src.prep_track.prep_track(reftrack_imp=reftrack_imp,
                                                    reg_smooth_opts=pars["reg_smooth_opts"],
                                                    stepsize_opts=pars["stepsize_opts"],
                                                    debug=debug,
                                                    min_width=imp_opts["min_track_width"])

    # ------------------------------------------------------------------------------------------------------------------
    # CALL OPTIMIZATION ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    alpha_opt = tph.opt_min_curv.opt_min_curv(reftrack=reftrack_interp,
                                                normvectors=normvec_normalized_interp,
                                                A=a_interp,
                                                kappa_bound=pars["veh_params"]["curvlim"],
                                                w_veh=safety_width,
                                                print_debug=debug,
                                                plot_debug=False)[0]

    # ------------------------------------------------------------------------------------------------------------------
    # INTERPOLATE SPLINES TO SMALL DISTANCES BETWEEN RACELINE POINTS ---------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    raceline_interp, a_opt, coeffs_x_opt, coeffs_y_opt, spline_inds_opt_interp, t_vals_opt_interp, s_points_opt_interp,\
        spline_lengths_opt, el_lengths_opt_interp = tph.create_raceline.\
        create_raceline(refline=reftrack_interp[:, :2],
                        normvectors=normvec_normalized_interp,
                        alpha=alpha_opt,
                        stepsize_interp=pars["stepsize_opts"]["stepsize_interp_after_opt"])

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATE HEADING AND CURVATURE ----------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # calculate heading and curvature (analytically)
    psi_vel_opt, kappa_opt = tph.calc_head_curv_an.\
        calc_head_curv_an(coeffs_x=coeffs_x_opt,
                          coeffs_y=coeffs_y_opt,
                          ind_spls=spline_inds_opt_interp,
                          t_spls=t_vals_opt_interp)

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATE VELOCITY AND ACCELERATION PROFILE ----------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    vx_profile_opt = tph.calc_vel_profile.\
        calc_vel_profile(ggv=ggv,
                            ax_max_machines=ax_max_machines,
                            v_max=pars["veh_params"]["v_max"],
                            kappa=kappa_opt,
                            el_lengths=el_lengths_opt_interp,
                            closed=True,
                            filt_window=pars["vel_calc_opts"]["vel_profile_conv_filt_window"],
                            dyn_model_exp=pars["vel_calc_opts"]["dyn_model_exp"],
                            drag_coeff=pars["veh_params"]["dragcoeff"],
                            m_veh=pars["veh_params"]["mass"])

    # calculate longitudinal acceleration profile
    vx_profile_opt_cl = np.append(vx_profile_opt, vx_profile_opt[0])
    ax_profile_opt = tph.calc_ax_profile.calc_ax_profile(vx_profile=vx_profile_opt_cl,
                                                         el_lengths=el_lengths_opt_interp,
                                                         eq_length_output=False)

    # calculate laptime
    t_profile_cl = tph.calc_t_profile.calc_t_profile(vx_profile=vx_profile_opt,
                                                     ax_profile=ax_profile_opt,
                                                     el_lengths=el_lengths_opt_interp)
    print("INFO: Estimated laptime: %.2fs" % t_profile_cl[-1])

    if plot_opts["racetraj_vel"] and plot:
        s_points = np.cumsum(el_lengths_opt_interp[:-1])
        s_points = np.insert(s_points, 0, 0.0)

        plt.plot(s_points, vx_profile_opt)
        plt.plot(s_points, ax_profile_opt)
        plt.plot(s_points, t_profile_cl[:-1])

        plt.grid()
        plt.xlabel("distance in m")
        plt.legend(["vx in m/s", "ax in m/s2", "t in s"])

        plt.show()

    # ------------------------------------------------------------------------------------------------------------------
    # DATA POSTPROCESSING ----------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # arrange data into one trajectory
    trajectory_opt = np.column_stack((s_points_opt_interp,
                                      raceline_interp,
                                      psi_vel_opt,
                                      kappa_opt,
                                      vx_profile_opt,
                                      ax_profile_opt))
    spline_data_opt = np.column_stack((spline_lengths_opt, coeffs_x_opt, coeffs_y_opt))

    # create a closed race trajectory array
    traj_race_cl = np.vstack((trajectory_opt, trajectory_opt[0, :]))
    traj_race_cl[-1, 0] = np.sum(spline_data_opt[:, 0])  # set correct length

    # print end time
    print("INFO: Runtime from import to final trajectory was %.2fs" % (time.perf_counter() - t_start))

    # ------------------------------------------------------------------------------------------------------------------
    # CHECK TRAJECTORY -------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    bound1, bound2 = helper_funcs_glob.src.check_traj.\
        check_traj(reftrack=reftrack_interp,
                   reftrack_normvec_normalized=normvec_normalized_interp,
                   length_veh=pars["veh_params"]["length"],
                   width_veh=pars["veh_params"]["width"],
                   debug=debug,
                   trajectory=trajectory_opt,
                   ggv=ggv,
                   ax_max_machines=ax_max_machines,
                   v_max=pars["veh_params"]["v_max"],
                   curvlim=pars["veh_params"]["curvlim"],
                   mass_veh=pars["veh_params"]["mass"],
                   dragcoeff=pars["veh_params"]["dragcoeff"])

    # ------------------------------------------------------------------------------------------------------------------
    # PLOT RESULTS -----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if plot:
        # get bound of imported map (for reference in final plot)
        bound1_imp = None
        bound2_imp = None

        if plot_opts["imported_bounds"]:
            # try to extract four times as many points as in the interpolated version (in order to hold more details)
            n_skip = max(int(reftrack_imp.shape[0] / (bound1.shape[0] * 4)), 1)

            _, _, _, normvec_imp = tph.calc_splines.calc_splines(path=np.vstack((reftrack_imp[::n_skip, 0:2],
                                                                                 reftrack_imp[0, 0:2])))

            bound1_imp = reftrack_imp[::n_skip, :2] + normvec_imp * np.expand_dims(reftrack_imp[::n_skip, 2], 1)
            bound2_imp = reftrack_imp[::n_skip, :2] - normvec_imp * np.expand_dims(reftrack_imp[::n_skip, 3], 1)

        # plot results
        helper_funcs_glob.src.result_plots.result_plots(plot_opts=plot_opts,
                                                        width_veh_opt=pars["optim_opts"]["width_opt"],
                                                        width_veh_real=pars["veh_params"]["width"],
                                                        refline=reftrack_interp[:, :2],
                                                        bound1_imp=bound1_imp,
                                                        bound2_imp=bound2_imp,
                                                        bound1_interp=bound1,
                                                        bound2_interp=bound2,
                                                        trajectory=trajectory_opt)

    return traj_race_cl, bound1, bound2, t_profile_cl[-1]  # also return estimated lap time
