from source import orbmrun
from datetime import timedelta, datetime
import math
import numpy as np
import pandas as pd
import os

global Fontsize
Fontsize = 30


# %% Calculate delta V to recover
def calc_recover(V_lowest_orbit, alt_drop, a_nom, thrust, drag, duty_cycle=1):
    r2 = a_nom
    r1 = a_nom - alt_drop
    t_maneuver = (
        V_lowest_orbit
        * (np.sqrt(2 * r2 / (r1 + r2)) - 1)
        * (sc_mass / (thrust * duty_cycle - drag))
    )
    del_V_recover = V_lowest_orbit * (np.sqrt(2 * r2 / (r1 + r2)) - 1) + (
        t_maneuver * drag / sc_mass
    )
    return t_maneuver, del_V_recover


def V_orbit(a):
    return math.sqrt(3.98e14 / a)


def plot_chemical_infeasible(df, attitude, altitude, contour_flag=False):
    """
    Plot to show that very difficult for chemical propulsion"""
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    df_reduced = df[df["Nom altitude (km)"] == altitude]

    # Plot heatmap, X as duty cycle, Y as thrust level, Z as the total maneuver time to return to operation, how about altitude?
    extent = (
        np.min(df_reduced["Duty cycle"]),
        np.max(df_reduced["Duty cycle"]),
        np.min(df_reduced["Thrust level (N)"]),
        np.max(df_reduced["Thrust level (N)"]),
    )
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=300)
    # plt.imshow(data,
    #     label=r"Max pitch angle ($\degree$)",
    #     linewidth=2,
    # )
    total_maneuver_time_heatmap = np.zeros(
        [
            df_reduced["Thrust level (N)"].unique().shape[0],
            df_reduced["Duty cycle"].unique().shape[0],
        ]
    )
    for i in range(total_maneuver_time_heatmap.shape[0]):
        for j in range(total_maneuver_time_heatmap.shape[1]):
            print(i * total_maneuver_time_heatmap.shape[1] + j)
            data_pt = df_reduced["Total maneuver time (hr)"][
                i * total_maneuver_time_heatmap.shape[1] + j
            ]
            if data_pt == "LOSS":
                data_pt = np.nan
            total_maneuver_time_heatmap[i][j] = data_pt
    total_maneuver_time_heatmap = total_maneuver_time_heatmap
    cmap = mpl.colormaps.get_cmap(
        "viridis"
    )  # viridis is the default colormap for imshow
    cmap.set_bad(color="red")

    c = ax.imshow(
        total_maneuver_time_heatmap,
        extent=extent,
        interpolation="nearest",
        aspect="auto",
        origin="lower",
        cmap=cmap,
    )
    cb = plt.colorbar(c, ax=ax)
    cb.ax.tick_params(labelsize=Fontsize)
    if contour_flag:
        X, Z = np.meshgrid(
            df_reduced["Duty cycle"].unique(), df_reduced["Thrust level (N)"].unique()
        )
        CS = ax.contour(
            X,
            Z,
            total_maneuver_time_heatmap,
            colors="r",
            levels=int(np.floor(X.shape[1] / 2)),
        )
        ax.clabel(CS, inline=True, fontsize=Fontsize, rightside_up=True)
    plt.xlabel(r"Duty cycle", fontsize=Fontsize)
    plt.ylabel(r"Thrust level (N)", fontsize=Fontsize)
    plt.title(
        rf"Total time (hours) to recover after 2 days, {altitude}km, chemical only. Greatest drag (mN): {np.min(df_reduced['Drag at lowest point (mN)']):.3f}. Final altitude (km): {np.min(df_reduced['Final altitude (km)']):.3f}",
        fontsize=Fontsize,
        wrap=True,
    )
    plt.grid()
    # plt.legend(fontsize=Fontsize * 0.7)
    plt.yticks(fontsize=Fontsize * 0.7)
    plt.xticks(fontsize=Fontsize * 0.7)
    # plt.show()
    plt.savefig(
        f"plots/{altitude}_km_safehold_recover_time_{attitude}.png",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    alts = [200, 220, 250, 300, 350]
    days_to_run_list = [2, 2, 3, 8, 8]

    thrust_list = np.arange(0.5, 4, 0.25)
    duty_cycle_input_list = np.arange(0.001, 1, 0.1)

    sc_drag_values = {
        "min_drag_area": {
            "area": 0.35,
            "sc_Cd": [3.84, 3.87, 4.233, 4.49, 4.578],
        },
        "max_yaw": {
            "area": 0.94,
            "sc_Cd": [2.70, 2.71, 3.05257, 3.25, 3.285],
        },
        "max_pitch": {
            "area": 4.32,
            "sc_Cd": [2.15, 2.15, 2.49, 2.66, 2.66],
        },
    }
    for attitude in sc_drag_values.keys():
        df = pd.DataFrame()
        columns = [
            "Nom altitude (km)",
            "Duty cycle",
            "Thrust level (N)",
            "Days of safe hold mode",
            "Final altitude (km)",
            "Change in mean semi major axis",
            "Drag at lowest point (mN)",
            "Total impulse (Ns)",
            "Total firing time (hr)",
            "Total duty cycles",
            "Total maneuver time (hr)",
        ]
        sc_area_d = sc_drag_values[attitude]["area"]
        for idx, alt in enumerate(alts):
            # %% User inputs
            alt_initial = alt
            days_to_run = days_to_run_list[idx]

            # %% Fixed case
            tstart = "1 Jan 2028 12:00:00.000"
            sc_Cd = sc_drag_values[attitude]["sc_Cd"][idx]
            orb_a = 6378 + alt_initial
            orb_e = 0.001
            orb_i = 10
            orb_R = 90
            orb_w = 45
            orb_m = -45
            maintenance_tolerance = 100
            maintenance_fro = False
            sc_mass = 200
            isp_min = 50
            isp_max = 50
            density_file = os.path.join("env", "DTM2020_weather_data_NEO.csv")

            # %% Preprocess some parameters
            datetime_format = "%d %b %Y %H:%M:%S.%f"
            tfinal = datetime.strftime(
                datetime.strptime(tstart, datetime_format)
                + timedelta(days=days_to_run),
                datetime_format,
            )
            # %% Run script
            epoch, alt, sma, dv, imp, final_drag = orbmrun.orbmrun(
                tstart,
                tfinal,
                sc_Cd,
                sc_area_d,
                orb_a,
                orb_e,
                orb_i,
                orb_R,
                orb_w,
                orb_m,
                maintenance_tolerance,
                maintenance_fro,
                sc_mass,
                isp_min,
                isp_max,
                density_file,
            )
            alt_drop = alt_initial - sma[-1] + 6378
            final_alt = alt_initial - alt_drop
            print(
                f"With inital altitude {alt_initial}km, final altitude {final_alt:.3f} km is reached, altitude drop of {alt_drop:.3f} km"
            )

            print(f"Drag value: {final_drag} N")

            del_V = np.zeros([len(duty_cycle_input_list) * len(thrust_list)])
            t_man = np.zeros([len(duty_cycle_input_list) * len(thrust_list)])
            total_impulse_array = np.zeros(
                [len(duty_cycle_input_list) * len(thrust_list)], dtype="object"
            )  # np.zeros(len(thrust_list))
            total_firing_time_array = np.zeros(
                [len(duty_cycle_input_list) * len(thrust_list)], dtype="object"
            )

            total_maneuver_time_array = np.zeros(
                [len(duty_cycle_input_list) * len(thrust_list)], dtype="object"
            )
            total_duty_cycles_array = np.zeros(
                [len(duty_cycle_input_list) * len(thrust_list)], dtype="object"
            )
            thrust_level_list = []
            duty_cycle_list = []
            duty_cycle_iterations = 0

            for count, thrust in enumerate(thrust_list):
                for count_duty_cycle, duty_cycle in enumerate(duty_cycle_input_list):
                    idx = (
                        len(duty_cycle_input_list) * (duty_cycle_iterations)
                    ) + count_duty_cycle
                    (
                        t_man[idx],
                        del_V[idx],
                    ) = calc_recover(
                        V_orbit(sma[-1] * 1e3),
                        1e3 * (alt_initial - sma[-1] + 6378),
                        orb_a * 1e3,
                        thrust,
                        final_drag,
                        duty_cycle,
                    )
                    duty_cycle_list.append(duty_cycle)
                    thrust_level_list.append(thrust)

                    total_maneuver_time_array[idx] = t_man[idx] / 3600
                    total_impulse_array[idx] = del_V[idx] * sc_mass
                    total_firing_time_array[idx] = (
                        total_impulse_array[idx] / thrust / 3600
                    )
                    total_duty_cycles_array[idx] = np.ceil(
                        total_maneuver_time_array[idx] / total_firing_time_array[idx]
                    )

                    if total_impulse_array[idx] < 0:
                        total_impulse_array[idx] = "LOSS"
                        total_firing_time_array[idx] = "LOSS"
                        total_maneuver_time_array[idx] = "LOSS"
                        total_duty_cycles_array[idx] = "LOSS"
                        print(f"{thrust}, {duty_cycle}")
                duty_cycle_iterations += 1
            print("___________________________________________________")
            temp_data = np.array(
                [
                    np.repeat(alt_initial, len(duty_cycle_list)),
                    duty_cycle_list,
                    thrust_level_list,
                    np.repeat(days_to_run, len(duty_cycle_list)),
                    np.repeat(final_alt, len(duty_cycle_list)),
                    np.repeat(alt_drop, len(duty_cycle_list)),
                    np.repeat(final_drag * 1000, len(duty_cycle_list)),
                ],
                dtype="object",
            )
            temp_data = np.vstack([temp_data, total_impulse_array])
            temp_data = np.vstack([temp_data, total_firing_time_array])
            temp_data = np.vstack([temp_data, total_duty_cycles_array])
            temp_data = np.vstack([temp_data, total_maneuver_time_array])

            temp_data = temp_data.transpose()
            temp_df = pd.DataFrame(
                columns=columns,
                data=temp_data,
            )

            df = pd.concat([df, temp_df])
        df.to_csv(f"safe_hold_{attitude}.csv")
        plot_chemical_infeasible(df, attitude, 220)
