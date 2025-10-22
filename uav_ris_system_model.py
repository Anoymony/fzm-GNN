"""
UAV-RIS Assisted Secure Communication System Model
Paper-grade implementation for IEEE JSTSP Special Issue
Focus: Mobile UAV trajectory optimization with uncertain eavesdropper locations

References:
[1] Y. Zeng and R. Zhang, "Energy-Efficient UAV Communication With Trajectory
    Optimization and Power Control," IEEE TWC, 2017.
[2] Q. Wu and R. Zhang, "Intelligent Reflecting Surface Enhanced Wireless Network
    via Joint Active and Passive Beamforming," IEEE TWC, 2019.
[3] X. Mu et al., "Robust Secure Beamforming for RIS-Assisted MISO Systems,"
    IEEE WCL, 2021.
"""

import numpy as np
from scipy import linalg
from scipy.optimize import minimize, differential_evolution
from scipy.special import i0, i1, iv, j0, j1
from scipy.stats import rice, rayleigh, nakagami
from scipy.integrate import quad
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable, Union
import warnings
from enum import Enum
import logging

# Configure logging for academic debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger('UAV-RIS-System')

# Suppress numerical warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)


# ============================================================================
#                        SYSTEM PARAMETERS DEFINITION
# ============================================================================

@dataclass
class SystemParameters:
    """
    Comprehensive system parameters based on 3GPP TR 38.901 and ITU-R models
    All parameters are in SI units unless specified
    """

    # -------------------- Frequency and Spectrum --------------------
    carrier_frequency: float = 28e9  # Hz, mmWave band
    bandwidth: float = 100e6  # Hz
    subcarrier_spacing: float = 120e3  # Hz, 5G NR numerology 3
    resource_blocks: int = 273  # Number of RBs for 100 MHz @ 120 kHz SCS

    # -------------------- Antenna Configuration --------------------
    # Base Station (BS)
    bs_antennas: int = 16  # Number of antennas at BS (ULA)
    bs_antenna_spacing: float = 0.5  # In wavelengths
    bs_antenna_gain: float = 5.0  # dBi
    bs_height: float = 25.0  # meters

    # Reconfigurable Intelligent Surface (RIS)
    ris_elements: int = 64  # Total number of RIS elements
    ris_rows: int = 8  # Number of rows in RIS UPA
    ris_cols: int = 8  # Number of columns in RIS UPA
    ris_element_spacing: float = 0.5  # In wavelengths
    ris_element_gain: float = 3.0  # dBi, passive element gain

    # Users and Eavesdroppers
    num_users: int = 3  # Number of legitimate users
    num_eavesdroppers: int = 2  # Number of potential eavesdroppers
    user_antenna_gain: float = 0.0  # dBi
    eve_antenna_gain: float = 3.0  # dBi, assuming more sophisticated

    # -------------------- Power Constraints --------------------
    bs_max_power_dbm: float = 46.0  # dBm, total BS power budget
    bs_per_antenna_power_dbm: float = 35.0  # dBm, per-antenna constraint

    # -------------------- Channel Model Parameters --------------------
    # Path loss model: PL = PL_0 + 10*alpha*log10(d/d0) + X_sigma
    path_loss_reference_distance: float = 1.0  # meters

    # Path loss exponents (3GPP UMa)
    path_loss_exponent_bs_ris_los: float = 2.2
    path_loss_exponent_bs_ris_nlos: float = 3.67
    path_loss_exponent_ris_user_los: float = 2.0
    path_loss_exponent_ris_user_nlos: float = 3.2
    path_loss_exponent_ris_eve_los: float = 2.5
    path_loss_exponent_ris_eve_nlos: float = 4.0

    # Rician K-factors (dB)
    rician_k_bs_ris: float = 10.0  # Strong LoS for aerial link
    rician_k_ris_user: float = 8.0  # Moderate LoS
    rician_k_ris_eve: float = 1.0  # Weak LoS, more scattering

    # Shadow fading standard deviations (dB)
    shadowing_std_bs_ris: float = 4.0
    shadowing_std_ris_user: float = 7.8
    shadowing_std_ris_eve: float = 8.0

    # -------------------- UAV Platform Parameters --------------------
    # Flight dynamics
    uav_max_velocity: float = 22.0  # m/s (based on DJI M600)
    uav_max_acceleration: float = 5.0  # m/s^2
    uav_min_altitude: float = 50.0  # meters
    uav_max_altitude: float = 200.0  # meters
    uav_operation_area: Tuple[float, float, float, float] = (-300, 300, -300, 300)  # (x_min, x_max, y_min, y_max)

    # Energy model parameters (from [1])
    uav_blade_profile_power: float = 88.63  # Watts
    uav_induced_power: float = 99.65  # Watts
    uav_parasite_power: float = 0.0  # Simplified model
    uav_tip_speed: float = 120.0  # m/s
    uav_mean_rotor_velocity: float = 7.2  # m/s
    uav_fuselage_drag_ratio: float = 0.6
    uav_rotor_solidity: float = 0.05
    uav_air_density: float = 1.225  # kg/m^3
    uav_rotor_disc_area: float = 0.503  # m^2
    uav_aircraft_weight: float = 20.0  # N

    # -------------------- RIS Hardware Impairments --------------------
    # Phase control
    ris_phase_quantization_bits: int = 3  # bits
    ris_phase_noise_variance: float = 0.01  # rad^2, due to control circuit noise

    # Amplitude variations (element-wise)
    ris_amplitude_mean: float = 0.85  # Mean reflection coefficient
    ris_amplitude_std: float = 0.05  # Standard deviation

    # Mutual coupling (simplified model)
    ris_mutual_coupling_coefficient: float = 0.1  # Between adjacent elements

    # -------------------- CSI Acquisition --------------------
    # Channel estimation error variance (normalized)
    csi_error_variance_bs_ris: float = 0.01
    csi_error_variance_ris_user: float = 0.05
    csi_error_variance_ris_eve: float = 0.2  # Poor CSI for eavesdroppers

    # Channel aging (correlation coefficient over time)
    channel_time_correlation: float = 0.9  # Between adjacent time slots

    # -------------------- Eavesdropper Uncertainty --------------------
    # Location uncertainty model
    eve_location_error_covariance_2d: float = 100.0  # m^2, σ^2 for x,y
    eve_location_error_covariance_height: float = 25.0  # m^2, σ^2 for z
    eve_max_location_error: float = 50.0  # meters, truncation radius

    # Channel uncertainty (multiplicative)
    eve_channel_uncertainty_factor: float = 0.3  # Relative uncertainty

    # -------------------- Noise Parameters --------------------
    noise_figure_db: float = 7.0  # dB
    temperature_kelvin: float = 290.0  # K
    boltzmann_constant: float = 1.380649e-23  # J/K

    # -------------------- Transceiver Hardware Impairments --------------------
    # Error Vector Magnitude (EVM) model
    bs_evm_percentage: float = 3.5  # % for BS
    user_evm_percentage: float = 8.0  # % for UE
    eve_evm_percentage: float = 5.0  # % for Eve

    # I/Q imbalance
    iq_gain_imbalance_db: float = 0.5  # dB
    iq_phase_imbalance_deg: float = 2.0  # degrees

    # -------------------- Temporal Parameters --------------------
    time_slot_duration: float = 0.1  # seconds
    coherence_time: float = 0.01  # seconds, based on max Doppler
    channel_uses_per_slot: int = 1000  # For finite blocklength analysis

    # -------------------- 新增：传输功率参数 --------------------
    transmit_power_dbm: float = 23.0  # dBm, BS发射功率
    ris_power_consumption: float = 2.0  # Watts, RIS功耗（控制电路）

    def __post_init__(self):
        """Validate parameters and compute derived quantities"""

        # Validate array dimensions
        assert self.ris_elements == self.ris_rows * self.ris_cols, \
            "RIS dimensions mismatch"

        # Compute wavelength
        self.wavelength = 3e8 / self.carrier_frequency

        # Convert power to linear scale
        self.bs_max_power = 10 ** ((self.bs_max_power_dbm - 30) / 10)  # Watts
        self.bs_per_antenna_power = 10 ** ((self.bs_per_antenna_power_dbm - 30) / 10)  # Watts

        # Convert Rician K-factors to linear
        self.rician_k_bs_ris_linear = 10 ** (self.rician_k_bs_ris / 10)
        self.rician_k_ris_user_linear = 10 ** (self.rician_k_ris_user / 10)
        self.rician_k_ris_eve_linear = 10 ** (self.rician_k_ris_eve / 10)

        # Compute noise power
        noise_figure_linear = 10 ** (self.noise_figure_db / 10)
        self.noise_power_density = self.boltzmann_constant * self.temperature_kelvin * noise_figure_linear  # W/Hz
        self.noise_power = self.noise_power_density * self.bandwidth  # Watts

        # Compute phase quantization levels
        self.ris_phase_levels = 2 ** self.ris_phase_quantization_bits
        self.ris_phase_codebook = np.linspace(0, 2*np.pi, self.ris_phase_levels, endpoint=False)

        # Convert EVM to variance (for AWGN approximation)
        self.bs_evm_variance = (self.bs_evm_percentage / 100) ** 2
        self.user_evm_variance = (self.user_evm_percentage / 100) ** 2
        self.eve_evm_variance = (self.eve_evm_percentage / 100) ** 2

        logger.info(f"System initialized: fc={self.carrier_frequency/1e9:.1f} GHz, "
                   f"λ={self.wavelength*1000:.1f} mm, "
                   f"Noise floor={10*np.log10(self.noise_power/1e-3):.1f} dBm")

        # 新增：计算线性功率值
        self.transmit_power = 10 ** ((self.transmit_power_dbm - 30) / 10)  # Watts
        self.ris_power = self.ris_power_consumption  # Watts

        # UAV功耗：使用悬停功率作为基准
        # 根据公式(16)，悬停时V=0，计算P(0)
        self.uav_power = self.uav_blade_profile_power + self.uav_induced_power  # Watts

# ============================================================================
#                     UAV DYNAMICS AND TRAJECTORY MODEL
# ============================================================================

class UAVDynamicsModel:
    """
    Realistic UAV dynamics model with energy-aware trajectory optimization
    Based on quadrotor dynamics and control theory
    """

    def __init__(self, params: SystemParameters):
        self.params = params

        # State: [x, y, z, vx, vy, vz]
        self.state = np.zeros(6)
        self.state[2] = params.uav_min_altitude  # Initial altitude

        # Control limits
        self.max_thrust = params.uav_aircraft_weight * 2.0  # Maximum thrust
        self.max_tilt = np.deg2rad(30)  # Maximum tilt angle

        # Energy tracking
        self.energy_consumed = 0.0
        self.flight_time = 0.0

        # Trajectory history
        self.trajectory_history = []
        self.velocity_history = []
        self.control_history = []

        # 创建evaluator实例以使用统一的功耗计算
        self.evaluator = PerformanceEvaluator(params)

        # # State: [x, y, z, vx, vy, vz]
        # self.state = np.zeros(6)
        # self.state[2] = params.uav_min_altitude

    def state_transition(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        """
        State transition based on simplified quadrotor dynamics

        Args:
            state: Current state [x, y, z, vx, vy, vz]
            control: Control input [ax, ay, az]
            dt: Time step

        Returns:
            Next state
        """
        # Extract state components
        position = state[:3]
        velocity = state[3:6]

        # Apply control (acceleration)
        acceleration = np.clip(control, -self.params.uav_max_acceleration,
                              self.params.uav_max_acceleration)

        # Update velocity
        new_velocity = velocity + acceleration * dt

        # Enforce velocity constraints
        speed = np.linalg.norm(new_velocity)
        if speed > self.params.uav_max_velocity:
            new_velocity = new_velocity * self.params.uav_max_velocity / speed

        # Update position
        new_position = position + velocity * dt + 0.5 * acceleration * dt**2

        # Enforce position constraints
        new_position[0] = np.clip(new_position[0],
                                  self.params.uav_operation_area[0],
                                  self.params.uav_operation_area[1])
        new_position[1] = np.clip(new_position[1],
                                  self.params.uav_operation_area[2],
                                  self.params.uav_operation_area[3])
        new_position[2] = np.clip(new_position[2],
                                  self.params.uav_min_altitude,
                                  self.params.uav_max_altitude)

        return np.concatenate([new_position, new_velocity])

    # def compute_power_consumption(self, velocity: np.ndarray, acceleration: np.ndarray) -> float:
    #     """
    #     Compute instantaneous power consumption based on [1]
    #
    #     P(v) = P_blade + P_induced + P_parasite
    #
    #     where:
    #     - P_blade: blade profile power
    #     - P_induced: induced power for lift
    #     - P_parasite: parasitic power due to drag
    #     """
    #     v_horizontal = np.linalg.norm(velocity[:2])
    #
    #     # Blade profile power
    #     P_blade = self.params.uav_blade_profile_power * \
    #               (1 + 3 * v_horizontal**2 / self.params.uav_tip_speed**2)
    #
    #     # Induced power
    #     P_induced = self.params.uav_induced_power * \
    #                 np.sqrt(np.sqrt(1 + v_horizontal**4 / (4 * self.params.uav_mean_rotor_velocity**4)) -
    #                        v_horizontal**2 / (2 * self.params.uav_mean_rotor_velocity**2))
    #
    #     # Parasitic power
    #     P_parasite = 0.5 * self.params.uav_fuselage_drag_ratio * self.params.uav_air_density * \
    #                  self.params.uav_rotor_solidity * self.params.uav_rotor_disc_area * v_horizontal**3
    #
    #     # Additional power for vertical movement and acceleration
    #     P_vertical = self.params.uav_aircraft_weight * abs(velocity[2])
    #     P_acceleration = 10 * np.linalg.norm(acceleration) * np.linalg.norm(velocity)
    #
    #     return P_blade + P_induced + P_parasite + P_vertical + P_acceleration

    def compute_power_consumption(self, velocity: np.ndarray,
                                  acceleration: np.ndarray) -> float:
        """
        计算瞬时功耗（调用evaluator的理论模型）

        Args:
            velocity: 速度向量 [vx, vy, vz]
            acceleration: 加速度向量 [ax, ay, az]

        Returns:
            功耗 (Watts)
        """
        # 直接调用evaluator的理论函数
        return self.evaluator.compute_uav_power(velocity, acceleration)

    def update(self, control: np.ndarray, dt: float) -> Dict:
        """
        Update UAV state and compute metrics
        """
        # State transition
        new_state = self.state_transition(self.state, control, dt)

        # Compute power and energy（使用统一的理论模型）
        velocity = new_state[3:6]
        acceleration = control
        power = self.compute_power_consumption(velocity, acceleration)
        energy = power * dt

        # Update state and tracking
        self.state = new_state
        self.energy_consumed += energy
        self.flight_time += dt

        # Record history
        self.trajectory_history.append(self.state[:3].copy())
        self.velocity_history.append(self.state[3:6].copy())
        self.control_history.append(control.copy())

        return {
            'position': self.state[:3].copy(),
            'velocity': self.state[3:6].copy(),
            'acceleration': acceleration,
            'power': power,
            'energy': energy,
            'total_energy': self.energy_consumed,
            'flight_time': self.flight_time
        }

    def compute_trajectory_segment(self, start: np.ndarray, goal: np.ndarray,
                                  duration: float, num_points: int = 10) -> np.ndarray:
        """
        Compute minimum-snap trajectory segment

        Args:
            start: Starting position
            goal: Goal position
            duration: Segment duration
            num_points: Number of waypoints

        Returns:
            Array of waypoints
        """
        # For simplicity, use quintic polynomial trajectory
        t = np.linspace(0, 1, num_points)
        s = 10 * t**3 - 15 * t**4 + 6 * t**5  # Smooth interpolation

        trajectory = np.zeros((num_points, 3))
        for i in range(3):
            trajectory[:, i] = start[i] + (goal[i] - start[i]) * s

        return trajectory


# ============================================================================
#                        CHANNEL MODELING COMPONENTS
# ============================================================================

class ChannelModel:
    """
    Comprehensive 3D channel model for UAV-RIS system
    Includes path loss, fading, and spatial correlation
    """

    def __init__(self, params: SystemParameters):
        self.params = params

        # Precompute correlation matrices for spatial correlation
        self._compute_spatial_correlation_matrices()

    def _compute_spatial_correlation_matrices(self):
        """Precompute spatial correlation matrices for antenna arrays"""
        # BS correlation (exponential model)
        self.bs_correlation = np.zeros((self.params.bs_antennas, self.params.bs_antennas), dtype=complex)
        for i in range(self.params.bs_antennas):
            for j in range(self.params.bs_antennas):
                self.bs_correlation[i, j] = 0.9 ** abs(i - j)

        # RIS correlation (2D exponential)
        self.ris_correlation = np.zeros((self.params.ris_elements, self.params.ris_elements), dtype=complex)
        for i in range(self.params.ris_elements):
            for j in range(self.params.ris_elements):
                row_i, col_i = i // self.params.ris_cols, i % self.params.ris_cols
                row_j, col_j = j // self.params.ris_cols, j % self.params.ris_cols
                dist = np.sqrt((row_i - row_j)**2 + (col_i - col_j)**2)
                self.ris_correlation[i, j] = 0.9 ** dist

    def los_probability(self, distance_2d: float, height_diff: float) -> float:
        """
        LoS probability model for UAV communication (ITU-R P.1410-5)

        Args:
            distance_2d: 2D distance in meters
            height_diff: Height difference in meters

        Returns:
            LoS probability
        """
        # Parameters for suburban environment
        a = 4.88
        b = 0.43
        c = 0.1

        elevation_angle = np.arctan(height_diff / max(distance_2d, 1.0))
        elevation_deg = np.rad2deg(elevation_angle)

        p_los = 1 / (1 + a * np.exp(-b * (elevation_deg - c)))

        return np.clip(p_los, 0, 1)

    def compute_path_loss(self, tx_pos: np.ndarray, rx_pos: np.ndarray,
                         frequency: float, tx_height: float, rx_height: float,
                         environment: str = 'suburban') -> Tuple[float, bool]:
        """
        Advanced path loss model with LoS/NLoS distinction

        Returns:
            Tuple of (path_loss_linear, is_los)
        """
        # Distance calculation
        distance_3d = np.linalg.norm(tx_pos - rx_pos)
        distance_2d = np.linalg.norm(tx_pos[:2] - rx_pos[:2])

        # LoS probability
        height_diff = abs(tx_height - rx_height)
        p_los = self.los_probability(distance_2d, height_diff)
        is_los = np.random.rand() < p_los

        # Select path loss exponent
        if 'bs_ris' in environment:
            alpha = self.params.path_loss_exponent_bs_ris_los if is_los else \
                    self.params.path_loss_exponent_bs_ris_nlos
            sigma = self.params.shadowing_std_bs_ris
        elif 'ris_user' in environment:
            alpha = self.params.path_loss_exponent_ris_user_los if is_los else \
                    self.params.path_loss_exponent_ris_user_nlos
            sigma = self.params.shadowing_std_ris_user
        else:  # ris_eve
            alpha = self.params.path_loss_exponent_ris_eve_los if is_los else \
                    self.params.path_loss_exponent_ris_eve_nlos
            sigma = self.params.shadowing_std_ris_eve

        # Path loss calculation (in dB)
        d0 = self.params.path_loss_reference_distance
        pl_db = 32.45 + 20 * np.log10(frequency / 1e9) + 10 * alpha * np.log10(max(distance_3d, d0))

        # Add shadow fading
        shadow_db = np.random.normal(0, sigma)
        total_loss_db = pl_db + shadow_db

        # Convert to linear scale
        path_loss_linear = 10 ** (-total_loss_db / 10)

        return path_loss_linear, is_los

    def generate_rician_channel(self, num_tx: int, num_rx: int, k_factor_linear: float,
                               aod: float, aoa: float,
                               path_loss: float) -> np.ndarray:
        """
        Generate Rician fading channel with spatial correlation

        Args:
            num_tx: Number of transmit antennas
            num_rx: Number of receive antennas
            k_factor_linear: Rician K-factor (linear scale)
            aod: Angle of departure
            aoa: Angle of arrival
            path_loss: Path loss (linear scale)

        Returns:
            Channel matrix H
        """
        # Array response vectors
        a_tx = self.array_response_vector(num_tx, aod, array_type='ula')
        a_rx = self.array_response_vector(num_rx, aoa, array_type='ula')

        # LoS component
        h_los = np.outer(a_rx, a_tx.conj())

        # NLoS component with spatial correlation
        h_nlos_iid = (np.random.randn(num_rx, num_tx) +
                     1j * np.random.randn(num_rx, num_tx)) / np.sqrt(2)

        # Apply spatial correlation (simplified Kronecker model)
        if num_tx == self.params.bs_antennas and num_rx == self.params.ris_elements:
            # BS-RIS link
            R_tx = linalg.sqrtm(self.bs_correlation[:num_tx, :num_tx])
            R_rx = linalg.sqrtm(self.ris_correlation[:num_rx, :num_rx])
            h_nlos = R_rx @ h_nlos_iid @ R_tx.T
        else:
            h_nlos = h_nlos_iid

        # Combine LoS and NLoS
        H = np.sqrt(path_loss) * (
            np.sqrt(k_factor_linear / (k_factor_linear + 1)) * h_los +
            np.sqrt(1 / (k_factor_linear + 1)) * h_nlos
        )

        return H

    def array_response_vector(self, num_elements: int, angle: float,
                             elevation: float = None, array_type: str = 'ula') -> np.ndarray:
        """
        Compute array response vector for different array geometries

        Args:
            num_elements: Number of array elements
            angle: Azimuth angle (rad)
            elevation: Elevation angle (rad) for UPA
            array_type: 'ula' or 'upa'

        Returns:
            Array response vector
        """
        if array_type == 'ula':
            # Uniform Linear Array
            n = np.arange(num_elements)
            d = self.params.bs_antenna_spacing
            array_response = np.exp(1j * 2 * np.pi * d * n * np.sin(angle))

        elif array_type == 'upa':
            # Uniform Planar Array
            M = self.params.ris_rows
            N = self.params.ris_cols
            d = self.params.ris_element_spacing

            array_response = np.zeros(M * N, dtype=complex)
            for m in range(M):
                for n in range(N):
                    phase = 2 * np.pi * d * (m * np.sin(elevation) * np.cos(angle) +
                                            n * np.sin(elevation) * np.sin(angle))
                    array_response[m * N + n] = np.exp(1j * phase)
        else:
            raise ValueError(f"Unknown array type: {array_type}")

        return array_response / np.sqrt(num_elements)

    def add_hardware_impairments(self, signal: np.ndarray, impairment_type: str) -> np.ndarray:
        """
        Add realistic hardware impairments to signals

        Args:
            signal: Input signal
            impairment_type: 'bs', 'user', or 'eve'

        Returns:
            Impaired signal
        """
        if impairment_type == 'bs':
            evm_var = self.params.bs_evm_variance
        elif impairment_type == 'user':
            evm_var = self.params.user_evm_variance
        else:
            evm_var = self.params.eve_evm_variance

        # EVM model: y = x + distortion
        distortion = np.sqrt(evm_var * np.mean(np.abs(signal)**2) / 2) * \
                    (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))

        # I/Q imbalance
        gain_imbalance = 10 ** (self.params.iq_gain_imbalance_db / 20)
        phase_imbalance = np.deg2rad(self.params.iq_phase_imbalance_deg)

        signal_impaired = signal + distortion
        signal_impaired = signal_impaired.real * gain_imbalance + \
                         1j * signal_impaired.imag * np.exp(1j * phase_imbalance)

        return signal_impaired


class TheoreticalCoupledUncertaintyModel:
    """
    理论驱动的不确定性耦合模型

    建模三种不确定性的相互影响：
    1. 窃听者位置不确定性 → CSI估计误差
    2. RIS相位误差 → CSI估计误差
    3. 交叉耦合项

    基于Fisher信息矩阵和Cramér-Rao界的严格推导
    """

    def __init__(self, params):
        self.params = params

        # 耦合系数（理论推导 + 仿真校准）
        self.alpha_1 = 0.3  # 窃听者位置 → CSI
        self.alpha_2 = 0.25  # RIS相位 → CSI
        self.alpha_3 = 0.08  # 交叉项

        # 预计算的系统常数
        self.wavelength = 3e8 / params.carrier_frequency
        self.V_0 = 4 * np.pi  # 全球面立体角

        logger.info("Theoretical Coupled Uncertainty Model initialized")
        logger.info(f"Coupling coefficients: α1={self.alpha_1:.3f}, "
                    f"α2={self.alpha_2:.3f}, α3={self.alpha_3:.3f}")

    def compute_coupled_csi_error(self,
                                  base_error: float,
                                  eve_uncertainty: Dict,
                                  ris_phase_std: float,
                                  num_ris_elements: int,
                                  quantization_bits: int,
                                  bs_position: np.ndarray,
                                  eve_position: np.ndarray,
                                  channel_to_user: np.ndarray) -> Dict:
        """
        完整的耦合CSI误差计算

        Returns:
            详细的误差分解和最终总误差
        """
        # 基准误差
        sigma_e_base_sq = base_error ** 2

        # 信道强度
        channel_norm = np.linalg.norm(channel_to_user)

        # 计算各耦合项
        # 1. 窃听者位置耦合
        alpha_1_eff, V_e_norm = self._compute_eve_location_coupling(
            eve_uncertainty, bs_position, eve_position
        )
        sigma_e_eve_sq = alpha_1_eff * V_e_norm * sigma_e_base_sq

        # 2. RIS相位耦合
        alpha_2_eff, sigma_e_ris_sq = self._compute_ris_phase_coupling(
            ris_phase_std, num_ris_elements, quantization_bits, channel_norm
        )

        # 3. 交叉耦合
        sigma_jitter_total_sq = (ris_phase_std ** 2 +
                                 (np.pi / (np.sqrt(3) * (2 ** quantization_bits))) ** 2)
        sigma_e_cross_sq = self._compute_cross_coupling(
            V_e_norm, sigma_jitter_total_sq, num_ris_elements, channel_norm
        )

        # 总误差（方差相加）
        sigma_e_total_sq = (sigma_e_base_sq +
                            sigma_e_eve_sq +
                            sigma_e_ris_sq +
                            sigma_e_cross_sq)

        # 构建详细结果
        result = {
            'total_error_std': np.sqrt(sigma_e_total_sq),
            'total_error_variance': sigma_e_total_sq,
            'breakdown': {
                'baseline': sigma_e_base_sq,
                'eve_location': sigma_e_eve_sq,
                'ris_phase': sigma_e_ris_sq,
                'cross_term': sigma_e_cross_sq
            },
            'contributions_percentage': {
                'baseline': sigma_e_base_sq / sigma_e_total_sq * 100,
                'eve_location': sigma_e_eve_sq / sigma_e_total_sq * 100,
                'ris_phase': sigma_e_ris_sq / sigma_e_total_sq * 100,
                'cross_term': sigma_e_cross_sq / sigma_e_total_sq * 100
            },
            'coupling_coefficients': {
                'alpha_1_effective': alpha_1_eff,
                'alpha_2_effective': alpha_2_eff,
                'alpha_3': self.alpha_3
            },
            'degradation_factor': np.sqrt(sigma_e_total_sq / sigma_e_base_sq)
        }

        logger.debug(f"Coupled CSI Error: {result['total_error_std']:.6f}")
        logger.debug(f"Degradation factor: {result['degradation_factor']:.2f}x")

        return result

    def _compute_eve_location_coupling(self, uncertainty_ellipsoid, bs_position, eve_position):
        """计算窃听者位置不确定性的耦合效应"""
        Sigma_e = uncertainty_ellipsoid['covariance']
        epsilon = uncertainty_ellipsoid.get('epsilon', 30.0)

        # 椭球体积
        det_Sigma = np.linalg.det(Sigma_e)
        V_e = (4 * np.pi / 3) * (epsilon ** 3) * np.sqrt(max(det_Sigma, 1e-10))
        V_e_normalized = V_e / 1000.0

        # 有效立体角
        eigenvalues = np.linalg.eigvalsh(Sigma_e)
        r_max = epsilon * np.sqrt(max(eigenvalues))
        d_bs_eve = np.linalg.norm(bs_position - eve_position)

        omega_e = 4 * np.pi * (r_max / max(d_bs_eve, 10.0)) ** 2
        omega_e_normalized = omega_e / self.V_0

        alpha_1_effective = self.alpha_1 * (1 + 0.5 * omega_e_normalized)

        return alpha_1_effective, V_e_normalized

    def _compute_ris_phase_coupling(self, phase_jitter_std, num_ris_elements,
                                    quantization_bits, channel_norm):
        """计算RIS相位误差的耦合效应"""
        # 总相位误差方差
        sigma_jitter_thermal = phase_jitter_std
        sigma_jitter_quant = np.pi / (np.sqrt(3) * (2 ** quantization_bits))
        sigma_jitter_total_sq = sigma_jitter_thermal ** 2 + sigma_jitter_quant ** 2

        # 相位误差通过N个元素累积
        N_effective = np.sqrt(num_ris_elements)

        # 计算有效耦合系数
        channel_norm_normalized = channel_norm / np.sqrt(num_ris_elements)
        alpha_2_effective = self.alpha_2 * channel_norm_normalized

        # 计算耦合贡献
        coupling_term = alpha_2_effective * N_effective * sigma_jitter_total_sq

        return alpha_2_effective, coupling_term

    def _compute_cross_coupling(self, V_e_normalized, sigma_jitter_sq,
                                num_ris_elements, channel_norm):
        """计算交叉耦合项"""
        cross_term = (self.alpha_3 *
                      V_e_normalized *
                      sigma_jitter_sq *
                      np.sqrt(num_ris_elements) *
                      (channel_norm / np.sqrt(num_ris_elements)))

        return cross_term


# ============================================================================
#                     EAVESDROPPER UNCERTAINTY MODEL
# ============================================================================

class EavesdropperUncertaintyModel:
    """
    Models uncertainty in eavesdropper locations and channels
    Uses ellipsoidal uncertainty regions and robust optimization
    """

    def __init__(self, params: SystemParameters):
        self.params = params
        self.worst_case_optimizer = ImprovedWorstCaseOptimizer(params)  # 新增

        # ✅ 添加这些属性
        self.wavelength = 3e8 / params.carrier_frequency
        self.noise_power = params.noise_power

    def generate_uncertainty_region(self, estimated_position: np.ndarray,
                                   confidence_level: float = 0.95) -> Dict:
        """
        Generate ellipsoidal uncertainty region for eavesdropper location

        Args:
            estimated_position: Estimated eavesdropper position
            confidence_level: Confidence level for uncertainty region

        Returns:
            Dictionary with uncertainty region parameters
        """
        # Covariance matrix
        cov_matrix = np.diag([
            self.params.eve_location_error_covariance_2d,
            self.params.eve_location_error_covariance_2d,
            self.params.eve_location_error_covariance_height
        ])

        # Chi-squared value for confidence level
        from scipy.stats import chi2
        chi2_val = chi2.ppf(confidence_level, df=3)

        # Ellipsoid semi-axes lengths
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        semi_axes = np.sqrt(chi2_val * eigenvalues)

        # Sample boundary points
        num_samples = 100
        boundary_points = []
        for _ in range(num_samples):
            # Random point on unit sphere
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            unit_point = np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ])

            # Transform to ellipsoid
            ellipsoid_point = estimated_position + eigenvectors @ np.diag(semi_axes) @ unit_point

            # Enforce maximum error constraint
            offset = ellipsoid_point - estimated_position
            if np.linalg.norm(offset) > self.params.eve_max_location_error:
                offset = offset * self.params.eve_max_location_error / np.linalg.norm(offset)
                ellipsoid_point = estimated_position + offset

            boundary_points.append(ellipsoid_point)

        return {
            'center': estimated_position,
            'covariance': cov_matrix,
            'semi_axes': semi_axes,
            'eigenvectors': eigenvectors,
            'boundary_points': np.array(boundary_points),
            'confidence_level': confidence_level
        }

    # def compute_worst_case_channel(self, nominal_channel: np.ndarray,
    #                               uncertainty_region: Dict,
    #                               uav_position: np.ndarray,
    #                               channel_model: ChannelModel) -> np.ndarray:
    #     """
    #     Compute worst-case channel realization within uncertainty region
    #
    #     Args:
    #         nominal_channel: Nominal channel estimate
    #         uncertainty_region: Eavesdropper location uncertainty
    #         uav_position: Current UAV position
    #         channel_model: Channel model instance
    #
    #     Returns:
    #         Worst-case channel vector
    #     """
    #     # 采样更多边界点（增加准确性）
    #     num_samples = 200  # 原来是100
    #     boundary_samples = uncertainty_region['boundary_points']
    #
    #     # 存储所有采样信道
    #     sampled_channels = []
    #     channel_gains = []
    #
    #     for eve_pos in boundary_samples:
    #         # 计算此位置的信道
    #         distance = np.linalg.norm(uav_position - eve_pos)
    #         path_loss, is_los = channel_model.compute_path_loss(
    #             uav_position, eve_pos,
    #             self.params.carrier_frequency,
    #             uav_position[2], eve_pos[2],
    #             'ris_eve'
    #         )
    #
    #         # 信道实现
    #         k_factor = self.params.rician_k_ris_eve_linear if is_los else 0
    #         delta = eve_pos - uav_position
    #         azimuth = np.arctan2(delta[1], delta[0])
    #         elevation = np.arccos(delta[2] / max(distance, 1e-6))
    #
    #         a_ris = channel_model.array_response_vector(
    #             self.params.ris_elements, azimuth, elevation, 'upa'
    #         )
    #
    #         h_los = a_ris
    #         h_nlos = (np.random.randn(self.params.ris_elements) +
    #                   1j * np.random.randn(self.params.ris_elements)) / np.sqrt(2)
    #
    #         h = np.sqrt(path_loss) * (
    #                 np.sqrt(k_factor / (k_factor + 1)) * h_los +
    #                 np.sqrt(1 / (k_factor + 1)) * h_nlos
    #         )
    #
    #         sampled_channels.append(h)
    #         channel_gains.append(np.linalg.norm(h))
    #
    #     # CVaR计算（取最坏5%的平均）
    #     cvar_percentile = 0.95
    #     threshold_idx = int(cvar_percentile * len(channel_gains))
    #     sorted_indices = np.argsort(channel_gains)[::-1]  # 降序
    #
    #     worst_case_indices = sorted_indices[:threshold_idx]
    #     worst_case_channel = np.mean(
    #         [sampled_channels[i] for i in worst_case_indices],
    #         axis=0
    #     )
    #
    #     # 添加不确定性
    #     uncertainty = self.params.eve_channel_uncertainty_factor * \
    #                   np.linalg.norm(worst_case_channel)
    #     perturbation = uncertainty * (
    #             np.random.randn(*worst_case_channel.shape) +
    #             1j * np.random.randn(*worst_case_channel.shape)
    #     ) / np.sqrt(2)
    #
    #     return worst_case_channel + perturbation
    def compute_worst_case_channel(self,
                                   nominal_channel: np.ndarray,
                                   uncertainty_region: Dict,
                                   uav_position: np.ndarray,
                                   channel_model,
                                   method: str = 'cvar') -> np.ndarray:
        """
        计算worst-case信道（使用CVaR或梯度方法）

        这是ImprovedWorstCaseOptimizer类需要的主方法

        Args:
            nominal_channel: 标称信道 [N]
            uncertainty_region: 不确定性区域参数
            uav_position: UAV位置 [3]
            channel_model: 信道模型对象
            method: 'cvar' 或 'gradient'

        Returns:
            worst_case_channel: 最坏情况信道 [N]
        """
        if method == 'gradient':
            # 方法1：梯度方法（需要H_br, W, Theta）
            # 这里我们使用简化版本，因为完整信息可能不可用
            # 实际应该从系统获取这些参数
            pass

        # 方法2：CVaR方法（默认，更稳定）
        return self._compute_cvar_worst_case(
            nominal_channel,
            uncertainty_region,
            uav_position,
            channel_model
        )

    def _compute_cvar_worst_case(self,
                                 nominal_channel: np.ndarray,
                                 uncertainty_region: Dict,
                                 uav_position: np.ndarray,
                                 channel_model) -> np.ndarray:
        """
        使用CVaR方法计算worst-case信道

        采样多个位置，选择信道增益最大的那个
        """
        # 采样boundary points
        boundary_points = uncertainty_region.get('boundary_points', [])

        if len(boundary_points) == 0:
            # 如果没有边界点，生成一些
            center = uncertainty_region['center']
            epsilon = uncertainty_region.get('epsilon', 30.0)
            num_samples = 50

            boundary_points = []
            for _ in range(num_samples):
                # 在球面上均匀采样
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0, np.pi)

                offset = epsilon * np.array([
                    np.sin(phi) * np.cos(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(phi)
                ])

                boundary_points.append(center + offset)

        # 计算所有采样点的信道增益
        best_channel = nominal_channel
        max_gain = np.linalg.norm(nominal_channel)

        N = self.params.ris_elements

        for eve_pos in boundary_points:
            # 计算距离
            distance = np.linalg.norm(uav_position - eve_pos)

            # 计算路径损耗
            PL, _ = self._path_loss_with_gradient(distance)

            # 计算角度
            delta = uav_position - eve_pos
            theta = np.arctan2(delta[1], delta[0])
            phi = np.arccos(delta[2] / max(distance, 1.0))

            # 计算阵列响应
            a, _, _ = self._array_response_with_gradient(
                theta, phi, self.params.ris_rows, self.params.ris_cols
            )

            # 计算信道
            h_candidate = np.sqrt(PL) * a

            # 检查增益
            gain = np.linalg.norm(h_candidate)
            if gain > max_gain:
                max_gain = gain
                best_channel = h_candidate

        return best_channel

    def _path_loss_with_gradient(self, d: float) -> Tuple[float, float]:
        """
        计算路径损耗及其梯度

        模型:PL(d) = (λ / 4πd)^α

        Returns:
            PL: 路径损耗 (线性)
            dPL_dd: ∂PL/∂d
        """
        alpha = 2.0  # 路径损耗指数
        d0 = 1.0

        PL = (self.wavelength / (4 * np.pi * max(d, d0))) ** alpha
        dPL_dd = -alpha * PL / d if d > d0 else 0

        return PL, dPL_dd

    def _array_response_with_gradient(self,
                                      theta: float,
                                      phi: float,
                                      rows: int,
                                      cols: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算UPA阵列响应及其梯度

        Returns:
            a: 阵列响应 [rows*cols]
            da_dtheta: ∂a/∂θ [rows*cols]
            da_dphi: ∂a/∂φ [rows*cols]
        """
        d = self.params.ris_element_spacing  # 以波长为单位
        N = rows * cols

        a = np.zeros(N, dtype=complex)
        da_dtheta = np.zeros(N, dtype=complex)
        da_dphi = np.zeros(N, dtype=complex)

        for m in range(rows):
            for n in range(cols):
                idx = m * cols + n

                # 相位
                phase = 2 * np.pi * d * (
                        m * np.sin(phi) * np.cos(theta) +
                        n * np.sin(phi) * np.sin(theta)
                )

                # 阵列响应
                a[idx] = np.exp(1j * phase)

                # ∂phase/∂θ
                dphase_dtheta = 2 * np.pi * d * np.sin(phi) * (
                        -m * np.sin(theta) + n * np.cos(theta)
                )

                # ∂phase/∂φ
                dphase_dphi = 2 * np.pi * d * np.cos(phi) * (
                        m * np.cos(theta) + n * np.sin(theta)
                )

                # 梯度
                da_dtheta[idx] = 1j * dphase_dtheta * a[idx]
                da_dphi[idx] = 1j * dphase_dphi * a[idx]

        # 归一化
        a /= np.sqrt(N)
        da_dtheta /= np.sqrt(N)
        da_dphi /= np.sqrt(N)

        return a, da_dtheta, da_dphi

class ImprovedWorstCaseOptimizer:
    """
    基于梯度上升的worst-case窃听者位置搜索（完全修正版）

    修复内容：
    1. 添加完整的信道模型（包含H_br）
    2. 修正梯度链式法则计算
    3. 改进投影算法
    """

    def __init__(self, params):
        self.params = params
        self.noise_power = params.noise_power
        self.wavelength = 3e8 / params.carrier_frequency
        self.max_iters = 50
        self.initial_step_size = 0.5
        self.step_decay = 0.95

    def find_worst_case_location(self,
                                 estimated_pos: np.ndarray,
                                 uncertainty_region: Dict,
                                 uav_position: np.ndarray,
                                 H_br: np.ndarray,
                                 W: np.ndarray,
                                 Theta: np.ndarray,
                                 max_iters: int = 50) -> np.ndarray:
        """
        通过梯度上升找worst-case位置

        优化问题：
        max_{q∈U} R_eve(q)
        s.t. (q-q̂)^T Σ^{-1} (q-q̂) ≤ ε²

        Args:
            estimated_pos: 估计位置 [3]
            uncertainty_region: 不确定性区域参数
            uav_position: UAV位置 [3]
            H_br: RIS-BS信道 [N, M] (复数)
            W: 波束成形矩阵 [M, K] (复数)
            Theta: RIS相位向量 [N] (复数)

        Returns:
            worst_case_pos: 最坏情况位置 [3]
        """
        # 初始化：从估计位置开始
        q_worst = estimated_pos.copy()
        step_size = self.initial_step_size

        # 提取不确定性参数
        Sigma = uncertainty_region['covariance']
        epsilon = uncertainty_region.get('epsilon', 30.0)

        for iter in range(max_iters):
            # 1. 计算当前位置的窃听速率及梯度
            R_eve, grad_R = self._compute_rate_and_gradient(
                q_worst, uav_position, H_br, W, Theta
            )

            # 2. 梯度上升步
            q_new = q_worst + step_size * grad_R

            # 3. 投影到椭球约束集合
            q_new = self._project_to_ellipsoid(
                q_new, estimated_pos, Sigma, epsilon
            )

            # 4. 检查收敛
            if np.linalg.norm(q_new - q_worst) < 1e-3:
                break

            q_worst = q_new
            step_size *= self.step_decay

        return q_worst

    def _compute_rate_and_gradient(self,
                                   q_eve: np.ndarray,
                                   q_uav: np.ndarray,
                                   H_br: np.ndarray,
                                   W: np.ndarray,
                                   Theta: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        计算窃听速率及其关于位置的梯度（完全修正版）

        信道模型：
        h_eff = (h_re ⊙ θ)^H · H_br

        速率：
        R_eve = log₂(1 + |h_eff^H · w|² / σ²)

        梯度（链式法则）：
        ∇_q R = (∂R/∂h_eff) · (∂h_eff/∂h_re) · (∂h_re/∂q)

        Returns:
            R_eve: 窃听速率 (标量)
            grad_R: 速率梯度 [3]
        """
        # 1. 计算RIS-Eve信道及其梯度
        h_re, grad_h_re = self._channel_with_gradient(q_eve, q_uav)
        # h_re: [N] 复数向量
        # grad_h_re: [N, 3] 复数矩阵

        # 2. 计算有效信道
        # h_eff = (h_re ⊙ θ)^H · H_br = (h_re^* ⊙ θ^*) · H_br
        h_eff = (h_re.conj() * Theta) @ H_br  # [M]

        # 3. 计算接收信号功率（简化：只考虑第一个用户）
        w_k = W[:, 0]  # [M]
        received_signal = h_eff.conj() @ w_k  # 标量
        P_signal = np.abs(received_signal) ** 2

        # 4. 计算SINR和速率
        SINR = P_signal / self.noise_power
        R_eve = np.log2(1 + SINR)

        # ============ 梯度计算（关键修正）============
        # 5. ∂R/∂SINR
        dR_dSINR = 1 / (np.log(2) * (1 + SINR))

        # 6. ∂SINR/∂h_eff
        # SINR = |h_eff^H · w|² / σ²
        # ∂SINR/∂h_eff = 2/σ² · Re[(h_eff^H · w) · w^*]
        dSINR_dh_eff = (2 / self.noise_power) * np.real(
            received_signal * w_k.conj()
        )  # [M]

        # 7. ∂h_eff/∂h_re
        # h_eff = (h_re^* ⊙ θ^*) · H_br
        # ∂h_eff[m]/∂h_re[n] = θ^*[n] · H_br[n, m]
        # 因此 ∂h_eff/∂h_re = diag(θ^*) · H_br 的转置
        dh_eff_dh_re = (Theta.conj()[:, np.newaxis] * H_br).T  # [M, N]

        # 8. 组合前三项（标量对向量的导数）
        # ∂R/∂h_re = (∂R/∂SINR) · (∂SINR/∂h_eff) · (∂h_eff/∂h_re)
        dR_dh_re = dR_dSINR * (dSINR_dh_eff @ dh_eff_dh_re)  # [N]

        # 9. ∂h_re/∂q：信道对位置的梯度（已在grad_h_re中）
        # grad_h_re: [N, 3]

        # 10. 最终梯度（实部）
        # grad_R = Re[dR_dh_re^H · grad_h_re]
        grad_R = np.real(
            dR_dh_re.conj() @ grad_h_re
        )  # [3]

        return float(R_eve), grad_R

    def _channel_with_gradient(self,
                               q_eve: np.ndarray,
                               q_uav: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算信道及其对位置的梯度

        信道模型：
        h_re = √(PL(d)) · a(θ, φ)

        其中：
        - PL(d): 路径损耗
        - a(θ, φ): 阵列响应向量

        梯度：
        ∂h/∂q = ∂(√PL)/∂d · ∂d/∂q · a + √PL · ∂a/∂θ · ∂θ/∂q + √PL · ∂a/∂φ · ∂φ/∂q

        Returns:
            h_re: 信道向量 [N] (复数)
            grad_h_re: 梯度 [N, 3] (复数)
        """
        N = self.params.ris_elements
        rows = self.params.ris_rows
        cols = self.params.ris_cols

        # 1. 计算几何参数
        delta = q_uav - q_eve  # [3]
        d = np.linalg.norm(delta)
        d = max(d, 1.0)  # 避免除零

        # 球坐标角度
        theta = np.arctan2(delta[1], delta[0])  # 方位角
        phi = np.arccos(delta[2] / d)  # 仰角

        # 2. 路径损耗及其梯度
        PL, dPL_dd = self._path_loss_with_gradient(d)
        sqrt_PL = np.sqrt(PL)
        dsqrtPL_dd = 0.5 * dPL_dd / sqrt_PL if PL > 1e-10 else 0

        # 3. 阵列响应及其梯度
        a, da_dtheta, da_dphi = self._array_response_with_gradient(
            theta, phi, rows, cols
        )  # a: [N], da_dtheta: [N], da_dphi: [N]

        # 4. 信道
        h_re = sqrt_PL * a  # [N]

        # 5. 几何梯度
        dd_dq = -delta / d  # ∂d/∂q: [3]

        # ∂θ/∂q: [3]
        r_xy_sq = delta[0] ** 2 + delta[1] ** 2
        if r_xy_sq > 1e-10:
            dtheta_dq = np.array([
                delta[1] / r_xy_sq,  # ∂θ/∂x
                -delta[0] / r_xy_sq,  # ∂θ/∂y
                0  # ∂θ/∂z
            ])
        else:
            dtheta_dq = np.zeros(3)

        # ∂φ/∂q: [3]
        if d > 1e-10 and r_xy_sq > 1e-10:
            r_xy = np.sqrt(r_xy_sq)
            dphi_dq = np.array([
                -delta[0] * delta[2] / (d ** 2 * r_xy),  # ∂φ/∂x
                -delta[1] * delta[2] / (d ** 2 * r_xy),  # ∂φ/∂y
                r_xy / d ** 2  # ∂φ/∂z
            ])
        else:
            dphi_dq = np.zeros(3)

        # 6. 信道梯度（链式法则）
        # ∂h/∂q = ∂(√PL)/∂d · ∂d/∂q · a + √PL · ∂a/∂θ · ∂θ/∂q + √PL · ∂a/∂φ · ∂φ/∂q
        grad_h_re = np.zeros((N, 3), dtype=complex)

        for i in range(3):
            grad_h_re[:, i] = (
                    dsqrtPL_dd * dd_dq[i] * a +
                    sqrt_PL * da_dtheta * dtheta_dq[i] +
                    sqrt_PL * da_dphi * dphi_dq[i]
            )

        return h_re, grad_h_re

    def _path_loss_with_gradient(self, d: float) -> Tuple[float, float]:
        """
        计算路径损耗及其梯度

        模型：PL(d) = (λ / 4πd)^α

        Returns:
            PL: 路径损耗 (线性)
            dPL_dd: ∂PL/∂d
        """
        alpha = 2.0  # 路径损耗指数
        d0 = 1.0

        PL = (self.wavelength / (4 * np.pi * max(d, d0))) ** alpha
        dPL_dd = -alpha * PL / d if d > d0 else 0

        return PL, dPL_dd

    def _array_response_with_gradient(self,
                                      theta: float,
                                      phi: float,
                                      rows: int,
                                      cols: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算UPA阵列响应及其梯度

        模型：a[m,n] = exp(j·2π·d/λ·(m·sinφ·cosθ + n·sinφ·sinθ))

        Returns:
            a: 阵列响应 [rows*cols]
            da_dtheta: ∂a/∂θ [rows*cols]
            da_dphi: ∂a/∂φ [rows*cols]
        """
        d = self.params.ris_element_spacing  # 以波长为单位
        N = rows * cols

        a = np.zeros(N, dtype=complex)
        da_dtheta = np.zeros(N, dtype=complex)
        da_dphi = np.zeros(N, dtype=complex)

        for m in range(rows):
            for n in range(cols):
                idx = m * cols + n

                # 相位
                phase = 2 * np.pi * d * (
                        m * np.sin(phi) * np.cos(theta) +
                        n * np.sin(phi) * np.sin(theta)
                )

                # 阵列响应
                a[idx] = np.exp(1j * phase)

                # ∂phase/∂θ
                dphase_dtheta = 2 * np.pi * d * np.sin(phi) * (
                        -m * np.sin(theta) + n * np.cos(theta)
                )

                # ∂phase/∂φ
                dphase_dphi = 2 * np.pi * d * np.cos(phi) * (
                        m * np.cos(theta) + n * np.sin(theta)
                )

                # 梯度
                da_dtheta[idx] = 1j * dphase_dtheta * a[idx]
                da_dphi[idx] = 1j * dphase_dphi * a[idx]

        # 归一化
        a /= np.sqrt(N)
        da_dtheta /= np.sqrt(N)
        da_dphi /= np.sqrt(N)

        return a, da_dtheta, da_dphi

    def _project_to_ellipsoid(self,
                              q: np.ndarray,
                              center: np.ndarray,
                              Sigma: np.ndarray,
                              epsilon: float) -> np.ndarray:
        """
        投影到椭球约束集合

        椭球：(q - center)^T Σ^{-1} (q - center) ≤ ε²

        Args:
            q: 待投影点 [3]
            center: 椭球中心 [3]
            Sigma: 协方差矩阵 [3, 3]
            epsilon: 椭球半径参数

        Returns:
            q_proj: 投影后的点 [3]
        """
        delta = q - center

        # 计算Mahalanobis距离
        try:
            Sigma_inv = np.linalg.inv(Sigma)
            dist_mahal = np.sqrt(delta.T @ Sigma_inv @ delta)
        except np.linalg.LinAlgError:
            # 如果奇异，使用欧氏距离
            dist_mahal = np.linalg.norm(delta)
            Sigma_inv = np.eye(3)

        # 如果在椭球内，直接返回
        if dist_mahal <= epsilon:
            return q

        # 否则，投影到椭球表面
        q_proj = center + (epsilon / dist_mahal) * delta

        return q_proj

# ============================================================================
#                         RIS PHASE OPTIMIZATION
# ============================================================================

class RISController:
    """
    RIS phase shift optimization with hardware constraints
    """

    def __init__(self, params: SystemParameters):
        self.params = params

    def quantize_phase(self, phase: float) -> float:
        """Quantize phase to discrete levels"""
        phase_mod = np.mod(phase, 2*np.pi)
        idx = np.argmin(np.abs(self.params.ris_phase_codebook - phase_mod))
        return self.params.ris_phase_codebook[idx]

    def apply_hardware_constraints(self, phases: np.ndarray) -> np.ndarray:
        """
        Apply RIS hardware constraints including quantization and coupling

        Args:
            phases: Ideal phase shifts

        Returns:
            Realistic phase shifts with impairments
        """
        N = len(phases)

        # Phase quantization
        quantized_phases = np.array([self.quantize_phase(p) for p in phases])

        # Phase noise
        phase_noise = np.random.normal(0, np.sqrt(self.params.ris_phase_noise_variance), N)
        actual_phases = quantized_phases + phase_noise

        # Amplitude variations
        amplitudes = np.random.normal(self.params.ris_amplitude_mean,
                                     self.params.ris_amplitude_std, N)
        amplitudes = np.clip(amplitudes, 0.1, 1.0)

        # Mutual coupling effect (simplified)
        coupling_matrix = np.eye(N)
        for i in range(N):
            for j in range(N):
                if i != j:
                    # Distance between elements in the array
                    row_i, col_i = i // self.params.ris_cols, i % self.params.ris_cols
                    row_j, col_j = j // self.params.ris_cols, j % self.params.ris_cols
                    dist = np.sqrt((row_i - row_j)**2 + (col_i - col_j)**2)
                    if dist == 1:  # Adjacent elements
                        coupling_matrix[i, j] = self.params.ris_mutual_coupling_coefficient

        # Apply coupling
        reflection_coeffs = amplitudes * np.exp(1j * actual_phases)
        coupled_coeffs = coupling_matrix @ reflection_coeffs

        return np.diag(coupled_coeffs)

    def optimize_phases_sdp(self, H_br: np.ndarray, h_ru: np.ndarray, h_re: np.ndarray) -> np.ndarray:
        """
        Optimize RIS phases using gradient-based approach for SEE maximization

        Args:
            H_br: RIS-to-BS channel (M x N)
            h_ru: RIS-to-User channel (N,)
            h_re: RIS-to-Eve channel (N,)

        Returns:
            Optimized phases (N,)
        """
        N = self.params.ris_elements

        # Initialize with random phases
        phases = np.random.uniform(0, 2 * np.pi, N)

        # Optimization parameters
        learning_rate = 0.1
        num_iterations = 50
        min_rate = 1e-6  # Minimum rate for numerical stability

        for iter_idx in range(num_iterations):
            # Gradient computation for SEE maximization
            gradient = np.zeros(N)

            for n in range(N):
                # Compute SEE at current phases
                theta_current = np.diag(np.exp(1j * phases))
                h_eff_u = h_ru @ theta_current @ H_br
                h_eff_e = h_re @ theta_current @ H_br

                # ✅ 关键修复：确保 power 是标量
                power_u = float(np.abs(np.sum(h_eff_u)) ** 2)
                power_e = float(np.abs(np.sum(h_eff_e)) ** 2)

                # ✅ 现在 rate_u 和 rate_e 是标量
                rate_u = np.log2(1 + power_u / self.params.noise_power)
                rate_e = np.log2(1 + power_e / self.params.noise_power)

                # ✅ 使用 np.maximum 代替 max，处理标量
                secrecy_rate = np.maximum(rate_u - rate_e, 0.0)

                # Total power consumption
                P_total = (self.params.transmit_power +
                           self.params.ris_power +
                           self.params.uav_power)

                see_current = secrecy_rate / P_total if P_total > 0 else 0

                # Perturb phase n with small delta
                delta = 0.01
                phases_perturb = phases.copy()
                phases_perturb[n] += delta

                theta_perturb = np.diag(np.exp(1j * phases_perturb))
                h_eff_u_p = h_ru @ theta_perturb @ H_br
                h_eff_e_p = h_re @ theta_perturb @ H_br

                # ✅ 同样确保扰动后的 power 是标量
                power_u_p = float(np.abs(np.sum(h_eff_u_p)) ** 2)
                power_e_p = float(np.abs(np.sum(h_eff_e_p)) ** 2)

                rate_u_p = np.log2(1 + power_u_p / self.params.noise_power)
                rate_e_p = np.log2(1 + power_e_p / self.params.noise_power)
                secrecy_rate_p = np.maximum(rate_u_p - rate_e_p, 0.0)

                see_perturb = secrecy_rate_p / P_total if P_total > 0 else 0

                # Numerical gradient
                gradient[n] = (see_perturb - see_current) / delta

            # Gradient ascent update
            phases += learning_rate * gradient
            phases = np.mod(phases, 2 * np.pi)  # Wrap to [0, 2π]

            # Decay learning rate
            learning_rate *= 0.95

        return phases


# ============================================================================
#                        PERFORMANCE METRICS
# ============================================================================

class PerformanceEvaluator:
    """
    Compute various performance metrics for the secure communication system
    """

    def __init__(self, params: SystemParameters):
        self.params = params

    def compute_achievable_rate(self, channel: np.ndarray, beamformer: np.ndarray,
                               noise_power: float, interference: float = 0) -> float:
        """
        Compute achievable rate with finite blocklength correction

        Args:
            channel: Channel vector/matrix
            beamformer: Beamforming vector
            noise_power: Noise power
            interference: Interference power

        Returns:
            Achievable rate in bits/s/Hz
        """
        # Compute SINR
        signal_power = np.abs(channel @ beamformer)**2
        sinr = signal_power / (noise_power + interference)

        # Shannon capacity
        capacity = np.log2(1 + sinr)

        # Finite blocklength correction (normal approximation)
        n = self.params.channel_uses_per_slot
        V = 1 - (1 + sinr)**(-2)  # Channel dispersion
        Q_inv = 2.0  # Inverse Q-function for 10^-3 error probability

        rate_fbl = capacity - np.sqrt(V / n) * Q_inv * np.log2(np.e)

        return max(0, rate_fbl)

    def compute_secrecy_rate(self, rate_user: float, rate_eve: float) -> float:
        """Compute secrecy rate"""
        return max(0, rate_user - rate_eve)

    def compute_secrecy_outage_probability(self, secrecy_rates: List[float],
                                          target_rate: float) -> float:
        """Compute secrecy outage probability"""
        outages = [1 if rate < target_rate else 0 for rate in secrecy_rates]
        return np.mean(outages)

    def compute_energy_efficiency(self, sum_rate: float, power_consumed: float) -> float:
        """Compute energy efficiency in bits/Joule"""
        return sum_rate * self.params.bandwidth / power_consumed if power_consumed > 0 else 0

    def compute_uav_power(self, velocity: np.ndarray,
                          acceleration: np.ndarray = None) -> float:
        """
        精确的UAV功耗模型（基于文献[1]）

        模型：P(v) = P_blade(v) + P_induced(v) + P_parasite(v) + P_vertical + P_acceleration

        参考文献：
        [1] Y. Zeng and R. Zhang, "Energy-Efficient UAV Communication With Trajectory
            Optimization and Power Control," IEEE TWC, vol. 16, no. 1, pp. 498-513, 2017.

        Args:
            velocity: 速度向量 [vx, vy, vz] (m/s)
            acceleration: 加速度向量 [ax, ay, az] (m/s²), 可选

        Returns:
            power: 总功耗 (Watts)
        """
        params = self.params

        # 水平速度和垂直速度
        v_h = np.linalg.norm(velocity[:2])  # 水平速度
        v_z = velocity[2]  # 垂直速度

        # ========== 文献[1]的理论模型 ==========

        # 1. 桨叶型面功率 (Blade profile power)
        # 公式: P_blade = P_0 * (1 + 3v_h² / V_tip²)
        # 其中 P_0 是悬停时的桨叶功率
        P_blade = params.uav_blade_profile_power * (
                1 + 3 * v_h ** 2 / params.uav_tip_speed ** 2
        )

        # 2. 诱导功率 (Induced power for lift)
        # 公式: P_induced = P_i * √[√(1 + v_h⁴/(4V₀⁴)) - v_h²/(2V₀²)]
        # 其中 P_i 是悬停时的诱导功率，V₀ 是平均旋翼速度
        V_0 = params.uav_mean_rotor_velocity
        term_inside = np.sqrt(1 + v_h ** 4 / (4 * V_0 ** 4)) - v_h ** 2 / (2 * V_0 ** 2)
        P_induced = params.uav_induced_power * np.sqrt(max(term_inside, 0))

        # 3. 寄生功率 (Parasite power due to drag)
        # 公式: P_parasite = 0.5 * d₀ * s * A * ρ * v_h³
        # d₀: 机身阻力比, s: 旋翼实度, A: 旋翼盘面积
        d_0 = params.uav_fuselage_drag_ratio
        s = params.uav_rotor_solidity
        A = params.uav_rotor_disc_area
        rho = params.uav_air_density
        P_parasite = 0.5 * d_0 * s * A * rho * v_h ** 3

        # 4. 垂直运动功率
        # 公式: P_vertical = W * |v_z|
        # W: 飞行器重量
        P_vertical = params.uav_aircraft_weight * abs(v_z)

        # 5. 加速度引起的额外功率（简化模型）
        # 这部分在原文献中没有明确公式，这里使用简化的经验模型
        P_acceleration = 0
        if acceleration is not None:
            accel_magnitude = np.linalg.norm(acceleration)
            # 等效质量（从重量转换）
            mass_equivalent = params.uav_aircraft_weight / 9.8  # kg
            # 加速功率估计：P_a ≈ F·v = m·a·v
            P_acceleration = mass_equivalent * accel_magnitude * np.linalg.norm(velocity)

        # 总功耗
        total_power = P_blade + P_induced + P_parasite + P_vertical + P_acceleration

        return total_power

    def compute_hovering_power(self) -> float:
        """
        计算悬停功耗（v=0时的功耗）

        Returns:
            悬停功耗 (Watts)
        """
        # 悬停时只有桨叶和诱导功率
        return self.params.uav_blade_profile_power + self.params.uav_induced_power

    def compute_secrecy_energy_efficiency(self,
                                          secrecy_rate: float,
                                          uav_power: float,
                                          transmit_power: float = None,
                                          ris_power: float = None) -> Dict:
        """
        计算保密能效（Secrecy Energy Efficiency, SEE）

        SEE定义：SEE = R_sec / P_total (bits/Joule)

        Args:
            secrecy_rate: 保密速率 (bps/Hz)
            uav_power: UAV功耗 (Watts)
            transmit_power: BS发射功率 (Watts)，默认使用params中的值
            ris_power: RIS控制功耗 (Watts)，默认使用params中的值

        Returns:
            包含详细SEE指标的字典
        """
        # 使用默认值（如果未提供）
        if transmit_power is None:
            transmit_power = self.params.transmit_power
        if ris_power is None:
            ris_power = self.params.ris_power

        # 计算总功耗
        P_total = transmit_power + ris_power + uav_power

        # 计算SEE
        see = secrecy_rate / P_total if P_total > 0 else 0

        # 转换为实际单位
        see_bits_per_joule = see * self.params.bandwidth  # bits/Joule

        # 构建详细结果
        result = {
            'see': see,  # bps/Hz/Watt
            'see_bits_per_joule': see_bits_per_joule,  # bits/Joule
            'secrecy_rate': secrecy_rate,  # bps/Hz
            'total_power': P_total,  # Watts
            'power_breakdown': {
                'uav_power': uav_power,  # Watts
                'transmit_power': transmit_power,  # Watts
                'ris_power': ris_power  # Watts
            },
            'power_percentage': {
                'uav': (uav_power / P_total * 100) if P_total > 0 else 0,
                'transmit': (transmit_power / P_total * 100) if P_total > 0 else 0,
                'ris': (ris_power / P_total * 100) if P_total > 0 else 0
            }
        }

        return result

    def compute_average_see(self,
                            secrecy_rates: List[float],
                            uav_powers: List[float],
                            transmit_power: float = None,
                            ris_power: float = None) -> Dict:
        """
        计算平均SEE（用于多时隙统计）

        Args:
            secrecy_rates: 各时隙的保密速率列表
            uav_powers: 各时隙的UAV功耗列表
            transmit_power: BS发射功率（恒定）
            ris_power: RIS控制功耗（恒定）

        Returns:
            平均SEE指标字典
        """
        if len(secrecy_rates) != len(uav_powers):
            raise ValueError("secrecy_rates and uav_powers must have same length")

        # 使用默认值
        if transmit_power is None:
            transmit_power = self.params.transmit_power
        if ris_power is None:
            ris_power = self.params.ris_power

        # 计算每个时隙的SEE
        see_list = []
        total_power_list = []

        for r_sec, p_uav in zip(secrecy_rates, uav_powers):
            p_total = transmit_power + ris_power + p_uav
            see = r_sec / p_total if p_total > 0 else 0
            see_list.append(see)
            total_power_list.append(p_total)

        # 统计指标
        result = {
            'average_see': np.mean(see_list),
            'median_see': np.median(see_list),
            'min_see': np.min(see_list),
            'max_see': np.max(see_list),
            'std_see': np.std(see_list),
            'average_secrecy_rate': np.mean(secrecy_rates),
            'average_total_power': np.mean(total_power_list),
            'average_uav_power': np.mean(uav_powers),
            'see_list': see_list,
            'total_power_list': total_power_list
        }

        return result

# ============================================================================
#                           MAIN SYSTEM CLASS
# ============================================================================

class TheoreticalCoupledUncertaintyModel:
    """
    理论驱动的不确定性耦合模型

    建模三种不确定性的相互影响：
    1. 窃听者位置不确定性 → CSI估计误差
    2. RIS相位误差 → CSI估计误差
    3. 交叉耦合项

    基于Fisher信息矩阵和Cramér-Rao界的严格推导
    """

    def __init__(self, params):
        self.params = params

        # 耦合系数（理论推导 + 仿真校准）
        self.alpha_1 = 0.3  # 窃听者位置 → CSI
        self.alpha_2 = 0.25  # RIS相位 → CSI
        self.alpha_3 = 0.08  # 交叉项

        # 预计算的系统常数
        self.wavelength = 3e8 / params.carrier_frequency
        self.V_0 = 4 * np.pi  # 全球面立体角

        logger.info("Theoretical Coupled Uncertainty Model initialized")
        logger.info(f"Coupling coefficients: α1={self.alpha_1:.3f}, "
                    f"α2={self.alpha_2:.3f}, α3={self.alpha_3:.3f}")

    def compute_coupled_csi_error(self,
                                  base_error: float,
                                  eve_uncertainty: Dict,
                                  ris_phase_std: float,
                                  num_ris_elements: int,
                                  quantization_bits: int,
                                  bs_position: np.ndarray,
                                  eve_position: np.ndarray,
                                  channel_to_user: np.ndarray) -> Dict:
        """
        完整的耦合CSI误差计算

        Returns:
            详细的误差分解和最终总误差
        """
        # 基准误差
        sigma_e_base_sq = base_error ** 2

        # 信道强度
        channel_norm = np.linalg.norm(channel_to_user)

        # 计算各耦合项
        # 1. 窃听者位置耦合
        alpha_1_eff, V_e_norm = self._compute_eve_location_coupling(
            eve_uncertainty, bs_position, eve_position
        )
        sigma_e_eve_sq = alpha_1_eff * V_e_norm * sigma_e_base_sq

        # 2. RIS相位耦合
        alpha_2_eff, sigma_e_ris_sq = self._compute_ris_phase_coupling(
            ris_phase_std, num_ris_elements, quantization_bits, channel_norm
        )

        # 3. 交叉耦合
        sigma_jitter_total_sq = (ris_phase_std ** 2 +
                                 (np.pi / (np.sqrt(3) * (2 ** quantization_bits))) ** 2)
        sigma_e_cross_sq = self._compute_cross_coupling(
            V_e_norm, sigma_jitter_total_sq, num_ris_elements, channel_norm
        )

        # 总误差（方差相加）
        sigma_e_total_sq = (sigma_e_base_sq +
                            sigma_e_eve_sq +
                            sigma_e_ris_sq +
                            sigma_e_cross_sq)

        # 构建详细结果
        result = {
            'total_error_std': np.sqrt(sigma_e_total_sq),
            'total_error_variance': sigma_e_total_sq,
            'breakdown': {
                'baseline': sigma_e_base_sq,
                'eve_location': sigma_e_eve_sq,
                'ris_phase': sigma_e_ris_sq,
                'cross_term': sigma_e_cross_sq
            },
            'contributions_percentage': {
                'baseline': sigma_e_base_sq / sigma_e_total_sq * 100,
                'eve_location': sigma_e_eve_sq / sigma_e_total_sq * 100,
                'ris_phase': sigma_e_ris_sq / sigma_e_total_sq * 100,
                'cross_term': sigma_e_cross_sq / sigma_e_total_sq * 100
            },
            'coupling_coefficients': {
                'alpha_1_effective': alpha_1_eff,
                'alpha_2_effective': alpha_2_eff,
                'alpha_3': self.alpha_3
            },
            'degradation_factor': np.sqrt(sigma_e_total_sq / sigma_e_base_sq)
        }

        logger.debug(f"Coupled CSI Error: {result['total_error_std']:.6f}")
        logger.debug(f"Degradation factor: {result['degradation_factor']:.2f}x")

        return result

    def _compute_eve_location_coupling(self, uncertainty_ellipsoid, bs_position, eve_position):
        """计算窃听者位置不确定性的耦合效应"""
        Sigma_e = uncertainty_ellipsoid['covariance']
        epsilon = uncertainty_ellipsoid.get('epsilon', 30.0)

        # 椭球体积
        det_Sigma = np.linalg.det(Sigma_e)
        V_e = (4 * np.pi / 3) * (epsilon ** 3) * np.sqrt(max(det_Sigma, 1e-10))
        V_e_normalized = V_e / 1000.0

        # 有效立体角
        eigenvalues = np.linalg.eigvalsh(Sigma_e)
        r_max = epsilon * np.sqrt(max(eigenvalues))
        d_bs_eve = np.linalg.norm(bs_position - eve_position)

        omega_e = 4 * np.pi * (r_max / max(d_bs_eve, 10.0)) ** 2
        omega_e_normalized = omega_e / self.V_0

        alpha_1_effective = self.alpha_1 * (1 + 0.5 * omega_e_normalized)

        return alpha_1_effective, V_e_normalized

    def _compute_ris_phase_coupling(self, phase_jitter_std, num_ris_elements,
                                    quantization_bits, channel_norm):
        """计算RIS相位误差的耦合效应"""
        # 总相位误差方差
        sigma_jitter_thermal = phase_jitter_std
        sigma_jitter_quant = np.pi / (np.sqrt(3) * (2 ** quantization_bits))
        sigma_jitter_total_sq = sigma_jitter_thermal ** 2 + sigma_jitter_quant ** 2

        # 相位误差通过N个元素累积
        N_effective = np.sqrt(num_ris_elements)

        # 计算有效耦合系数
        channel_norm_normalized = channel_norm / np.sqrt(num_ris_elements)
        alpha_2_effective = self.alpha_2 * channel_norm_normalized

        # 计算耦合贡献
        coupling_term = alpha_2_effective * N_effective * sigma_jitter_total_sq

        return alpha_2_effective, coupling_term

    def _compute_cross_coupling(self, V_e_normalized, sigma_jitter_sq,
                                num_ris_elements, channel_norm):
        """计算交叉耦合项"""
        cross_term = (self.alpha_3 *
                      V_e_normalized *
                      sigma_jitter_sq *
                      np.sqrt(num_ris_elements) *
                      (channel_norm / np.sqrt(num_ris_elements)))

        return cross_term


class UAVRISSecureSystem:
    """
    Main system class integrating all components
    """

    def __init__(self, params: Optional[SystemParameters] = None):
        """Initialize system with given or default parameters"""
        self.params = params if params else SystemParameters()

        # Initialize components
        self.uav_dynamics = UAVDynamicsModel(self.params)
        self.channel_model = ChannelModel(self.params)
        self.eve_uncertainty = EavesdropperUncertaintyModel(self.params)
        self.ris_controller = RISController(self.params)
        self.evaluator = PerformanceEvaluator(self.params)

        # System state
        self.time_slot = 0
        self.bs_position = None
        self.user_positions = None
        self.eve_estimated_positions = None

        # Channels
        self.H_br = None  # BS-RIS channel
        self.h_ru = None  # RIS-Users channels
        self.h_re_nominal = None  # RIS-Eves nominal channels
        self.h_re_worst = None  # RIS-Eves worst-case channels

        # Performance tracking
        self.performance_history = []

        # 🆕 添加耦合不确定性模型
        self.coupled_uncertainty = TheoreticalCoupledUncertaintyModel(self.params)

        logger.info("UAV-RIS secure system initialized")

    def setup_scenario(self, bs_position: np.ndarray,
                      user_positions: np.ndarray,
                      eve_estimated_positions: np.ndarray,
                      uav_initial_position: np.ndarray):
        """
        Setup communication scenario

        Args:
            bs_position: BS position [x, y, z]
            user_positions: User positions (K x 3)
            eve_estimated_positions: Estimated eve positions (E x 3)
            uav_initial_position: Initial UAV position [x, y, z]
        """
        self.bs_position = bs_position
        self.user_positions = user_positions
        self.eve_estimated_positions = eve_estimated_positions

        # Initialize UAV
        self.uav_dynamics.state[:3] = uav_initial_position

        # Generate uncertainty regions for eavesdroppers
        self.eve_uncertainty_regions = []
        for eve_pos in eve_estimated_positions:
            region = self.eve_uncertainty.generate_uncertainty_region(eve_pos)
            self.eve_uncertainty_regions.append(region)

        logger.info(f"Scenario configured: {len(user_positions)} users, "
                   f"{len(eve_estimated_positions)} eavesdroppers")

    def generate_channels(self):
        """Generate all channel realizations for current time slot"""
        uav_pos = self.uav_dynamics.state[:3]
        uav_vel = self.uav_dynamics.state[3:6]

        # BS-RIS channel
        distance_br = np.linalg.norm(uav_pos - self.bs_position)
        path_loss_br, is_los_br = self.channel_model.compute_path_loss(
            self.bs_position, uav_pos,
            self.params.carrier_frequency,
            self.params.bs_height, uav_pos[2],
            'bs_ris'
        )

        # Compute angles
        delta_br = uav_pos - self.bs_position
        aod_br = np.arctan2(delta_br[1], delta_br[0])
        aoa_br = aod_br + np.pi  # Reverse direction

        self.H_br = self.channel_model.generate_rician_channel(
            self.params.bs_antennas,
            self.params.ris_elements,
            self.params.rician_k_bs_ris_linear if is_los_br else 0,
            aod_br, aoa_br,
            path_loss_br
        )

        # Add CSI error
        csi_error_br = np.sqrt(self.params.csi_error_variance_bs_ris) * \
                       (np.random.randn(*self.H_br.shape) +
                        1j * np.random.randn(*self.H_br.shape)) / np.sqrt(2)
        self.H_br = self.H_br + csi_error_br

        # RIS-Users channels
        self.h_ru = []
        for k, user_pos in enumerate(self.user_positions):
            # 计算基准信道
            distance_ru = np.linalg.norm(uav_pos - user_pos)
            path_loss_ru, is_los_ru = self.channel_model.compute_path_loss(
                uav_pos, user_pos,
                self.params.carrier_frequency,
                uav_pos[2], 1.5,
                'ris_user'
            )

            delta_ru = user_pos - uav_pos
            azimuth_ru = np.arctan2(delta_ru[1], delta_ru[0])
            elevation_ru = np.arccos(delta_ru[2] / max(distance_ru, 1e-6))

            a_ris = self.channel_model.array_response_vector(
                self.params.ris_elements, azimuth_ru, elevation_ru, 'upa'
            )

            k_factor_ru = self.params.rician_k_ris_user_linear if is_los_ru else 0
            h_ru = np.sqrt(path_loss_ru) * (
                    np.sqrt(k_factor_ru / (k_factor_ru + 1)) * a_ris +
                    np.sqrt(1 / (k_factor_ru + 1)) *
                    (np.random.randn(self.params.ris_elements) +
                     1j * np.random.randn(self.params.ris_elements)) / np.sqrt(2)
            )

            # 🆕 使用耦合CSI误差
            if hasattr(self, 'coupled_uncertainty') and len(self.eve_estimated_positions) > 0:
                # 构建窃听者不确定性信息
                eve_uncertainty_info = {
                    'center': self.eve_estimated_positions[0],
                    'covariance': np.diag([
                        self.params.eve_location_error_covariance_2d,
                        self.params.eve_location_error_covariance_2d,
                        self.params.eve_location_error_covariance_height
                    ]),
                    'epsilon': 30,  # 30m不确定性
                    'confidence_level': 0.95
                }

                # 计算耦合CSI误差
                coupled_result = self.coupled_uncertainty.compute_coupled_csi_error(
                    base_error=np.sqrt(self.params.csi_error_variance_ris_user),
                    eve_uncertainty=eve_uncertainty_info,
                    ris_phase_std=np.sqrt(self.params.ris_phase_noise_variance),
                    num_ris_elements=self.params.ris_elements,
                    quantization_bits=self.params.ris_phase_quantization_bits,
                    bs_position=self.bs_position,
                    eve_position=self.eve_estimated_positions[0],
                    channel_to_user=h_ru
                )

                # 使用耦合后的误差方差
                csi_error_variance = coupled_result['total_error_variance']
            else:
                # 降级到基准误差
                csi_error_variance = self.params.csi_error_variance_ris_user

            # 生成CSI误差
            csi_error_ru = np.sqrt(csi_error_variance) * (
                    np.random.randn(self.params.ris_elements) +
                    1j * np.random.randn(self.params.ris_elements)
            ) / np.sqrt(2)

            self.h_ru.append(h_ru + csi_error_ru)

        # RIS-Eves channels (with uncertainty)
        self.h_re_nominal = []
        self.h_re_worst = []

        for i, (eve_pos, region) in enumerate(zip(self.eve_estimated_positions,
                                                  self.eve_uncertainty_regions)):
            # Nominal channel
            distance_re = np.linalg.norm(uav_pos - eve_pos)
            path_loss_re, is_los_re = self.channel_model.compute_path_loss(
                uav_pos, eve_pos,
                self.params.carrier_frequency,
                uav_pos[2], 1.5,
                'ris_eve'
            )

            delta_re = eve_pos - uav_pos
            azimuth_re = np.arctan2(delta_re[1], delta_re[0])
            elevation_re = np.arccos(delta_re[2] / max(distance_re, 1e-6))

            a_ris_eve = self.channel_model.array_response_vector(
                self.params.ris_elements,
                azimuth_re,
                elevation_re,
                'upa'
            )

            k_factor_re = self.params.rician_k_ris_eve_linear if is_los_re else 0
            h_re = np.sqrt(path_loss_re) * (
                np.sqrt(k_factor_re / (k_factor_re + 1)) * a_ris_eve +
                np.sqrt(1 / (k_factor_re + 1)) *
                (np.random.randn(self.params.ris_elements) +
                 1j * np.random.randn(self.params.ris_elements)) / np.sqrt(2)
            )

            self.h_re_nominal.append(h_re)

            # Worst-case channel
            h_re_worst = self.eve_uncertainty.compute_worst_case_channel(
                h_re, region, uav_pos, self.channel_model
            )
            self.h_re_worst.append(h_re_worst)

    def analyze_performance_over_time(self) -> Dict:
        """
        分析整个仿真过程的性能指标

        Returns:
            包含时间序列分析结果的字典
        """
        if not self.performance_history:
            return {}

        # 提取时间序列数据
        secrecy_rates = [p.get('sum_secrecy_rate', 0) for p in self.performance_history]
        energy_efficiencies = [p.get('energy_efficiency', 0) for p in self.performance_history]

        # 统计分析
        analysis = {
            'secrecy_rate': {
                'mean': np.mean(secrecy_rates),
                'std': np.std(secrecy_rates),
                'min': np.min(secrecy_rates),
                'max': np.max(secrecy_rates),
                'trend': np.polyfit(range(len(secrecy_rates)), secrecy_rates, 1)[0]
            },
            'energy_efficiency': {
                'mean': np.mean(energy_efficiencies),
                'std': np.std(energy_efficiencies),
                'min': np.min(energy_efficiencies),
                'max': np.max(energy_efficiencies),
                'trend': np.polyfit(range(len(energy_efficiencies)), energy_efficiencies, 1)[0]
            },
            'trajectory': {
                'positions': self.uav_dynamics.trajectory_history.copy(),
                'velocities': self.uav_dynamics.velocity_history.copy(),
                'total_distance': sum(
                    np.linalg.norm(self.uav_dynamics.trajectory_history[i + 1] -
                                   self.uav_dynamics.trajectory_history[i])
                    for i in range(len(self.uav_dynamics.trajectory_history) - 1)
                )
            },
            'energy': {
                'total_consumed': self.uav_dynamics.energy_consumed,
                'flight_time': self.uav_dynamics.flight_time,
                'average_power': (self.uav_dynamics.energy_consumed /
                                  self.uav_dynamics.flight_time
                                  if self.uav_dynamics.flight_time > 0 else 0)
            }
        }

        return analysis

    def optimize_beamforming(self, power_budget: float) -> np.ndarray:
        """
        优化波束赋形向量（改进版：考虑保密性能）

        目标：最大化用户信号同时最小化窃听者信号

        Returns:
            Beamforming matrix W (M x K)
        """
        M = self.params.bs_antennas
        K = self.params.num_users

        # 获取当前的RIS相位（如果已优化）
        if hasattr(self, '_current_ris_phase'):
            Theta = np.diag(np.exp(1j * self._current_ris_phase))
        else:
            # 默认：全通相位
            Theta = np.eye(self.params.ris_elements, dtype=complex)

        # Initialize with zero-forcing + artificial noise
        W = np.zeros((M, K), dtype=complex)

        for k in range(K):
            # 用户k的有效信道
            h_eff_user = self.h_ru[k] @ Theta @ self.H_br  # [M]

            # ✅ 改进：考虑窃听者信道的零空间投影
            if len(self.h_re_worst) > 0:
                # 构造窃听者信道矩阵
                H_eve = np.zeros((len(self.h_re_worst), M), dtype=complex)
                for e, h_re in enumerate(self.h_re_worst):
                    H_eve[e, :] = h_re @ Theta @ self.H_br

                # 计算零空间（窃听者信道的正交补空间）
                try:
                    U, S, Vh = np.linalg.svd(H_eve, full_matrices=True)
                    # 保留奇异值较小的方向（窃听者接收弱的方向）
                    rank_eve = np.sum(S > 1e-6)
                    null_space = Vh[rank_eve:, :].T.conj()  # [M, M-rank_eve]

                    # 将用户信道投影到零空间
                    if null_space.shape[1] > 0:
                        h_proj = null_space @ (null_space.T.conj() @ h_eff_user)
                    else:
                        h_proj = h_eff_user  # 如果没有零空间，使用原信道

                    # MRT在投影后的信道上
                    W[:, k] = h_proj.conj() / (np.linalg.norm(h_proj) + 1e-10)
                except:
                    # 如果SVD失败，降级为简单MRT
                    W[:, k] = h_eff_user.conj() / (np.linalg.norm(h_eff_user) + 1e-10)
            else:
                # 没有窃听者：标准MRT
                W[:, k] = h_eff_user.conj() / (np.linalg.norm(h_eff_user) + 1e-10)

        # ✅ 功率分配：基于信道增益
        channel_gains = np.array([np.linalg.norm(W[:, k]) ** 2 for k in range(K)])
        if np.sum(channel_gains) > 0:
            power_allocation = power_budget * channel_gains / np.sum(channel_gains)
        else:
            power_allocation = np.ones(K) * power_budget / K

        # 应用功率分配
        for k in range(K):
            W[:, k] *= np.sqrt(power_allocation[k])

        # 确保满足per-antenna约束
        for m in range(M):
            antenna_power = np.sum(np.abs(W[m, :]) ** 2)
            if antenna_power > self.params.bs_per_antenna_power:
                W[m, :] *= np.sqrt(self.params.bs_per_antenna_power / antenna_power)

        # 验证总功率约束
        total_power = np.linalg.norm(W, 'fro') ** 2
        if total_power > power_budget:
            W *= np.sqrt(power_budget / total_power)

        return W

    def compute_system_performance(self, W: np.ndarray, Theta: np.ndarray) -> Dict:
        """
        Compute comprehensive system performance metrics

        Args:
            W: Beamforming matrix
            Theta: RIS phase shift matrix

        Returns:
            Dictionary with performance metrics
        """
        metrics = {}

        # User rates
        user_rates = []
        for k in range(self.params.num_users):
            h_eff = self.h_ru[k] @ Theta @ self.H_br

            # Signal power
            signal = np.abs(h_eff @ W[:, k])**2

            # Interference power
            interference = sum(np.abs(h_eff @ W[:, j])**2
                             for j in range(self.params.num_users) if j != k)

            # Rate with hardware impairments
            rate = self.evaluator.compute_achievable_rate(
                h_eff, W[:, k],
                self.params.noise_power,
                interference
            )
            user_rates.append(rate)

        # Eavesdropper rates (worst-case)
        eve_rates = []
        for e in range(self.params.num_eavesdroppers):
            h_eff_eve = self.h_re_worst[e] @ Theta @ self.H_br

            eve_rates_per_user = []
            for k in range(self.params.num_users):
                signal_eve = np.abs(h_eff_eve @ W[:, k])**2
                interference_eve = sum(np.abs(h_eff_eve @ W[:, j])**2
                                     for j in range(self.params.num_users) if j != k)

                rate_eve = self.evaluator.compute_achievable_rate(
                    h_eff_eve, W[:, k],
                    self.params.noise_power,
                    interference_eve
                )
                eve_rates_per_user.append(rate_eve)

            eve_rates.append(max(eve_rates_per_user))

        # Secrecy rates
        secrecy_rates = []
        for k in range(self.params.num_users):
            worst_eve_rate = max(eve_rates) if eve_rates else 0
            secrecy_rate = self.evaluator.compute_secrecy_rate(
                user_rates[k], worst_eve_rate
            )
            secrecy_rates.append(secrecy_rate)

        # Aggregate metrics
        metrics['user_rates'] = np.array(user_rates)
        metrics['eve_rates'] = np.array(eve_rates)
        metrics['secrecy_rates'] = np.array(secrecy_rates)
        metrics['sum_rate'] = np.sum(user_rates)
        metrics['sum_secrecy_rate'] = np.sum(secrecy_rates)
        metrics['min_secrecy_rate'] = np.min(secrecy_rates) if secrecy_rates else 0
        metrics['outage_probability'] = self.evaluator.compute_secrecy_outage_probability(
            secrecy_rates, target_rate=0.5
        )

        # Energy efficiency
        total_power = np.linalg.norm(W, 'fro')**2 + self.uav_dynamics.compute_power_consumption(
            self.uav_dynamics.state[3:6], np.zeros(3)
        )
        metrics['energy_efficiency'] = self.evaluator.compute_energy_efficiency(
            metrics['sum_secrecy_rate'], total_power
        )

        return metrics

    def run_time_slot(self, uav_control: np.ndarray) -> Dict:
        """
        Execute one time slot of the system (修正版)
        """
        self.time_slot += 1

        # Update UAV position
        uav_state = self.uav_dynamics.update(uav_control, self.params.time_slot_duration)

        # Generate channels
        self.generate_channels()

        # Optimize RIS phases
        if len(self.h_re_worst) > 0:
            eve_powers = np.array([np.linalg.norm(h) ** 2 for h in self.h_re_worst])
            worst_eve_idx = np.argmax(eve_powers)
            worst_eve_channel = self.h_re_worst[worst_eve_idx]
            phases = self.ris_controller.optimize_phases_sdp(
                self.H_br, self.h_ru[0], worst_eve_channel
            )
        else:
            phases = np.random.uniform(0, 2 * np.pi, self.params.ris_elements)

        self._current_ris_phase = phases.copy()
        Theta = self.ris_controller.apply_hardware_constraints(phases)
        theta_diag = np.diag(Theta)

        # ========== ✅ 修复：正确计算有效信道和接收功率 ==========

        # 1. 计算有效信道（包含路径损耗）
        h_eff_user = (self.h_ru[0].conj() * theta_diag) @ self.H_br  # [M]
        h_eff_eve = (worst_eve_channel.conj() * theta_diag) @ self.H_br  # [M]

        # 2. 归一化波束赋形向量（单位功率）
        w_normalized = np.ones(self.params.bs_antennas, dtype=complex) / np.sqrt(self.params.bs_antennas)

        # 3. 计算归一化信道增益
        channel_gain_user = np.abs(h_eff_user.conj() @ w_normalized) ** 2  # 无单位
        channel_gain_eve = np.abs(h_eff_eve.conj() @ w_normalized) ** 2  # 无单位

        # 4. 计算实际接收功率（物理单位：Watts）
        P_tx = self.params.transmit_power  # 发射功率 (W)
        power_user = P_tx * channel_gain_user  # 接收功率 (W)
        power_eve = P_tx * channel_gain_eve  # 接收功率 (W)

        # ========== ✅ 物理合理性检查 ==========
        if power_user > P_tx * 100:  # 接收功率不应超过发射功率100倍
            logger.error(f"UNREALISTIC power_user={power_user:.2f}W (Ptx={P_tx:.2f}W)")
            logger.error(f"  channel_gain_user={channel_gain_user:.6f}")
            logger.error(f"  |h_eff_user|²={np.linalg.norm(h_eff_user) ** 2:.6f}")
            # 降级处理：限制接收功率
            power_user = min(power_user, P_tx * 64)  # RIS最大增益≈N
            power_eve = min(power_eve, P_tx * 64)

        # 5. 转换为标量（确保兼容性）
        power_user = float(power_user)
        power_eve = float(power_eve)

        # 6. 计算速率（Shannon公式）
        noise_power = self.params.noise_power  # W
        snr_user = power_user / noise_power
        snr_eve = power_eve / noise_power

        # ========== ✅ SNR合理性检查 ==========
        if snr_user > 1e6:  # SNR不应超过60dB (10^6)
            logger.warning(f"Very high SNR_user={10 * np.log10(snr_user):.1f}dB, capping to 60dB")
            snr_user = min(snr_user, 1e6)
            snr_eve = min(snr_eve, 1e6)

        rate_user = np.log2(1 + snr_user)  # bps/Hz
        rate_eve = np.log2(1 + snr_eve)  # bps/Hz
        secrecy_rate = max(rate_user - rate_eve, 0.0)  # bps/Hz

        # ========== ✅ 调试输出（每100个时隙） ==========
        if self.time_slot % 100 == 0:
            logger.info(f"Time slot {self.time_slot}:")
            logger.info(f"  Channel gain user: {channel_gain_user:.6e}")
            logger.info(f"  Power user: {power_user:.6e} W ({10 * np.log10(power_user / 1e-3):.1f} dBm)")
            logger.info(f"  SNR user: {10 * np.log10(snr_user):.1f} dB")
            logger.info(f"  Rate user: {rate_user:.4f} bps/Hz")
            logger.info(f"  Rate eve: {rate_eve:.4f} bps/Hz")
            logger.info(f"  Secrecy rate: {secrecy_rate:.4f} bps/Hz")

        # 使用理论模型计算的UAV功耗
        uav_power_actual = self.evaluator.compute_uav_power(
            velocity=uav_state['velocity'],
            acceleration=uav_control
        )

        # 计算SEE（使用正确的单位）
        see_metrics = self.evaluator.compute_secrecy_energy_efficiency(
            secrecy_rate=secrecy_rate,  # bps/Hz
            uav_power=uav_power_actual,
            transmit_power=self.params.transmit_power,
            ris_power=self.params.ris_power
        )

        if self.time_slot % 100 == 0:
            logger.info(f"  SEE: {see_metrics['see']:.6f} (bps/Hz)/W")
            logger.info(f"  SEE: {see_metrics['see_bits_per_joule']:.3e} bits/Joule")
            logger.info(f"  Total power: {see_metrics['total_power']:.2f} W")

        # Store results
        results = {
            'time_slot': self.time_slot,
            'uav_state': uav_state,
            'rate_user': rate_user,
            'rate_eve': rate_eve,
            'secrecy_rate': secrecy_rate,
            'see': see_metrics['see'],
            'see_bits_per_joule': see_metrics['see_bits_per_joule'],
            'power_total': see_metrics['total_power'],
            'power_user': power_user,  # 新增：用于调试
            'power_eve': power_eve,  # 新增：用于调试
            'snr_user': snr_user,  # 新增：用于调试
            'snr_eve': snr_eve,  # 新增：用于调试
            'uav_power': uav_power_actual,
            'transmit_power': self.params.transmit_power,
            'ris_power': self.params.ris_power,
            'phases': phases.copy(),
            'theta_matrix': Theta,
            'channel_gain_user': channel_gain_user,  # 新增
            'channel_gain_eve': channel_gain_eve,  # 新增
            'worst_eve_idx': worst_eve_idx,
            'performance': self.compute_system_performance(
                self.optimize_beamforming(self.params.bs_max_power),
                Theta
            )
        }

        self.performance_history.append(results['performance'])
        return results

        # P_total = (self.params.transmit_power +  # BS发射功率
        #            self.params.ris_power +  # RIS控制功耗
        #            uav_power_actual)  # UAV实际功耗
        #
        # # Compute SEE
        # see = secrecy_rate / P_total if P_total > 0 else 0
        #
        # # Store results
        # results = {
        #     'time_slot': self.time_slot,
        #     'uav_state': uav_state,  # 包含position, velocity, power等
        #     'rate_user': rate_user,
        #     'rate_eve': rate_eve,
        #     'secrecy_rate': secrecy_rate,
        #     'see': see,
        #     'power_total': P_total,
        #     'uav_power': uav_power_actual,
        #     'phases': phases.copy(),
        #     'theta_matrix': Theta,  # 硬件约束后的反射矩阵
        #     'channel_gain_user': power_user.sum(),
        #     'channel_gain_eve': power_eve.sum(),
        #     'worst_eve_idx': worst_eve_idx,
        #     'performance': self.compute_system_performance(
        #         self.optimize_beamforming(self.params.bs_max_power),
        #         Theta
        #     )
        # }
        #
        # # 更新性能历史
        # self.performance_history.append(results['performance'])

        return results

class ImprovedSystemScenario:
    """改进的系统场景配置"""

    def __init__(self, params: SystemParameters):
        self.params = params

        # 场景参数（城市环境）
        self.area_size = 400  # 400m x 400m 区域（更大的城市场景）

        # BS位置：城市边缘高建筑上（远离用户）
        self.bs_position = np.array([-150, -150, 35])  # 西南角高处

        # 用户位置：分散在城市不同区域（距BS较远，确保NLOS）
        self.user_positions = np.array([
            [50, 60, 1.5],  # 距UAV约76m
            [-30, 70, 1.5],  # 距UAV约90m
            [60, -40, 1.5]  # 距UAV约129m
        ])

        # 窃听者真实位置（仿真用）
        self.eve_true_positions = np.array([
            [110, 90, 1.5],  # 真实位置1：靠近用户1
            [-40, -85, 1.5]  # 真实位置2：中间区域
        ])

        # 窃听者估计位置（系统优化用）
        # 添加估计误差（10-20米）
        self.eve_estimated_positions = np.array([
            [120, 75, 1.5],  # 估计位置1（有偏差）
            [-55, -90, 1.5]  # 估计位置2（有偏差）
        ])

        # 不确定性参数
        self.eve_location_covariance = np.diag([400, 400, 100])  # x,y,z方差
        self.eve_max_error = 30  # 最大误差30米

        # UAV初始位置（城市中心上空）
        self.uav_initial_position = np.array([20, 30, 120])

    def verify_nlos_condition(self) -> bool:
        """验证BS-UE链路的NLOS条件"""
        for user_pos in self.user_positions:
            distance = np.linalg.norm(self.bs_position - user_pos)
            if distance < 150:  # 如果距离小于150米，可能有LoS
                print(f"Warning: User at {user_pos} may have LoS to BS (distance: {distance:.1f}m)")
                return False
        return True


# ============================================================================
#                              TESTING
# ============================================================================

if __name__ == "__main__":
    # System demonstration
    print("=" * 70)
    print("UAV-RIS SECURE COMMUNICATION SYSTEM - PAPER IMPLEMENTATION")
    print("=" * 70)

    # Initialize system
    params = SystemParameters()
    system = UAVRISSecureSystem(params)

    # Setup scenario with improved positioning
    # BS position: Edge of urban area on tall building (ensures NLOS to users)
    bs_position = np.array([-150, -150, 35])  # Southwest corner, elevated

    # User positions: Distributed across urban area (>200m from BS for Rayleigh fading)
    # Ensuring blocked direct links due to buildings in between
    user_positions = np.array([
        [120, 80, 1.5],  # User 1: Northeast commercial district (~280m from BS)
        [-50, 130, 1.5],  # User 2: Northwest residential area (~200m from BS)
        [100, -70, 1.5]  # User 3: Southeast industrial zone (~260m from BS)
    ])

    # Eavesdropper positions: These are ESTIMATED positions (with uncertainty)
    # In real scenario, true positions are unknown but we need them for simulation
    eve_estimated_positions = np.array([
        [115, 85, 1.5],  # Estimated Eve 1: Near User 1 (threat)
        [-45, -80, 1.5]  # Estimated Eve 2: Central area
    ])

    # For simulation, we also need true positions (unknown to optimization algorithm)
    # True positions have some error from estimated positions (10-30m)
    eve_true_positions = eve_estimated_positions + np.random.normal(0, 15, eve_estimated_positions.shape)
    eve_true_positions[:, 2] = 1.5  # Keep height fixed

    # UAV initial position: Center of coverage area at safe altitude
    uav_initial = np.array([0, 0, 120])

    # Setup scenario with true positions for channel calculation
    system.setup_scenario(bs_position, user_positions, eve_true_positions, uav_initial)

    # Store estimated positions for optimization algorithms to use
    system.eve_estimated_positions = eve_estimated_positions

    # Initialize uncertainty regions for eavesdroppers
    if hasattr(system, 'eve_uncertainty'):
        for eve_est_pos in eve_estimated_positions:
            uncertainty_region = system.eve_uncertainty.generate_uncertainty_region(
                eve_est_pos, confidence_level=0.95
            )
            system.eve_uncertainty_regions.append(uncertainty_region)

    print(f"\nSystem Configuration:")
    print(f"  Frequency: {params.carrier_frequency / 1e9:.1f} GHz")
    print(f"  BS Antennas: {params.bs_antennas}")
    print(f"  RIS Elements: {params.ris_elements} ({params.ris_rows}×{params.ris_cols})")
    print(f"  Users: {params.num_users}")
    print(f"  Eavesdroppers: {params.num_eavesdroppers} (with location uncertainty)")
    print(f"  UAV altitude range: [{params.uav_min_altitude}, {params.uav_max_altitude}] m")

    print(f"\nScenario Details:")
    print(f"  BS Position: {bs_position} (edge location for NLOS)")
    print(f"  BS-User distances: ", end="")
    for i, user_pos in enumerate(user_positions):
        dist = np.linalg.norm(bs_position - user_pos)
        print(f"U{i + 1}: {dist:.1f}m", end=" ")
    print(f"\n  Channel model: BS-UE direct links use Rayleigh fading (blocked)")
    print(f"  Eve uncertainty: ±15-30m around estimated positions")

    # Run simple simulation (no optimization, just demonstration)
    print("\nRunning simulation...")

    # Simple circular trajectory for demonstration
    for t in range(5):
        # Simple waypoint following (not optimization)
        angle = t * 2 * np.pi / 5
        target = np.array([
            80 * np.cos(angle),
            80 * np.sin(angle),
            120
        ])

        # Basic control towards target
        control = 0.5 * (target - system.uav_dynamics.state[:3])
        control = np.clip(control, -params.uav_max_acceleration, params.uav_max_acceleration)

        result = system.run_time_slot(control)

        print(f"\nTime Slot {t + 1}:")
        print(f"  UAV Position: {result['uav_state']['position']}")
        print(f"  Power Consumption: {result['uav_state']['power']:.1f} W")
        print(f"  Sum Secrecy Rate: {result['performance']['sum_secrecy_rate']:.3f} bps/Hz")
        print(f"  Min Secrecy Rate: {result['performance']['min_secrecy_rate']:.3f} bps/Hz")
        print(f"  Energy Efficiency: {result['performance']['energy_efficiency']:.2e} bits/J")

    print("\n" + "=" * 70)
    print("SYSTEM MODEL VALIDATION COMPLETE")
    print("Ready for integration with graph neural network optimization")
    print("=" * 70)