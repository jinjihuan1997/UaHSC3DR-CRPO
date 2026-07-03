"""Simple A2G channel model for UAV-to-ground links."""
import csv
import math


def dbm_to_w(dbm):
    return 10.0 ** ((float(dbm) - 30.0) / 10.0)


def thermal_noise_dbm(bandwidth_hz, noise_figure_db):
    return -174.0 + 10.0 * math.log10(float(bandwidth_hz)) + float(noise_figure_db)


def load_trajectory_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty trajectory csv: {path}")
    return rows


class A2GChannel:
    """Al-Hourani-style A2G path loss with log-normal shadowing."""

    def __init__(
        self,
        fc_hz=900e6,
        bandwidth_hz=1e6,
        ptx_dbm=20.0,
        noise_figure_db=7.0,
        shadow_sigma_db=3.0,
        los_a=9.61,
        los_b=0.16,
        eta_los_db=1.0,
        eta_nlos_db=20.0,
    ):
        self.fc_hz = float(fc_hz)
        self.bandwidth_hz = float(bandwidth_hz)
        self.ptx_dbm = float(ptx_dbm)
        self.ptx_w = dbm_to_w(self.ptx_dbm)
        self.noise_figure_db = float(noise_figure_db)
        self.shadow_sigma_db = float(shadow_sigma_db)
        self.los_a = float(los_a)
        self.los_b = float(los_b)
        self.eta_los_db = float(eta_los_db)
        self.eta_nlos_db = float(eta_nlos_db)
        self.noise_dbm = thermal_noise_dbm(self.bandwidth_hz, self.noise_figure_db)

    def snr_from_row(self, row, rng):
        ux = float(row["uav_x_global_m"])
        uy = float(row["uav_y_global_m"])
        uz = float(row["uav_z_global_m"])
        vx = float(row.get("vehicle_x_m", 0.0))
        vy = float(row.get("vehicle_y_m", 0.0))
        vz = float(row.get("vehicle_z_m", 1.5))

        dx = ux - vx
        dy = uy - vy
        dz = uz - vz
        horizontal = math.hypot(dx, dy)
        distance = max(math.sqrt(horizontal * horizontal + dz * dz), 1e-6)
        elevation_deg = math.degrees(math.atan2(max(dz, 0.0), max(horizontal, 1e-6)))

        c = 299792458.0
        fspl_db = 20.0 * math.log10(4.0 * math.pi * distance * self.fc_hz / c)
        p_los = 1.0 / (1.0 + self.los_a * math.exp(-self.los_b * (elevation_deg - self.los_a)))
        excess_db = p_los * self.eta_los_db + (1.0 - p_los) * self.eta_nlos_db
        shadow_db = float(rng.normal(0.0, self.shadow_sigma_db)) if self.shadow_sigma_db > 0.0 else 0.0
        path_loss_db = fspl_db + excess_db + shadow_db
        snr_db = self.ptx_dbm - path_loss_db - self.noise_dbm

        return {
            "snr_db": snr_db,
            "path_loss_db": path_loss_db,
            "shadow_db": shadow_db,
            "distance_m": distance,
            "elevation_angle_deg": elevation_deg,
            "p_los": p_los,
            "noise_dbm": self.noise_dbm,
            "ptx_dbm": self.ptx_dbm,
            "ptx_w": self.ptx_w,
        }
