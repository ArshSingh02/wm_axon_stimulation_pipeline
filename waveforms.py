from typing import Optional, Tuple, Callable
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d


class WaveformType:
    """Enumeration for waveform types."""
    CUSTOM = 1
    BIPHASIC = 2
    MONOPHASIC = 3
    RECTANGULAR = 4
    MST_BIPHASIC = 5


class Waveform:
    def __init__(self, waveform_type: WaveformType, path: Optional[str] = None):
        self.waveform_type = waveform_type
        self.time = None
        self.e_field_magnitude = None

        if waveform_type == WaveformType.CUSTOM:
            if path is None:
                raise IOError("Path to waveform file needed for custom waveform.")
            tms_wave = loadmat(path)
            self.time = tms_wave["time"].ravel()
            self.e_field_magnitude = tms_wave["e_mag"].ravel()
        elif waveform_type in {WaveformType.BIPHASIC, WaveformType.MONOPHASIC}:
            tms_waves = loadmat("/hpc/home/asb113/Desktop/WM Stimulation/Waveforms/TMSwaves.mat")
            self.time = tms_waves["tm"].ravel()
            if waveform_type == WaveformType.BIPHASIC:
                self.e_field_magnitude = tms_waves["Erec_b"].ravel()
            elif waveform_type == WaveformType.MONOPHASIC:
                self.e_field_magnitude = tms_waves["Erec_m"].ravel()
        elif waveform_type == WaveformType.MST_BIPHASIC:
            mst_wave_path = "/hpc/home/asb113/Desktop/WM Stimulation/Waveforms/MagVenture_MST_Twin.tsv"
            mst_wave = np.loadtxt(mst_wave_path, delimiter="\t", skiprows=1)
            self.time = mst_wave[:, 0] * 1000  # convert s to ms
            self.e_field_magnitude = mst_wave[:, 1]

    def load_waveform(self, simulation_time_step: float, stimulation_delay: float, simulation_duration: float) -> Callable[[float], float]:
        sample_factor = max(1, int(simulation_time_step / np.mean(np.diff(self.time))))
        sim_time = self.time[::sample_factor]
        sim_e_field = np.append(self.e_field_magnitude[::sample_factor], 0)
        waveform_duration = sim_time[-1]

        if stimulation_delay >= simulation_time_step:
            pre_time = np.arange(0, stimulation_delay, simulation_time_step)
            sim_time = np.concatenate((pre_time, sim_time + stimulation_delay))
            sim_e_field = np.concatenate((np.zeros_like(pre_time), sim_e_field))

        sim_time = np.append(
            sim_time,
            np.arange(sim_time[-1] + simulation_time_step, simulation_duration + simulation_time_step, simulation_time_step)
        )
        sim_e_field = np.pad(sim_e_field, (0, len(sim_time) - len(sim_e_field)), constant_values=0)

        return interp1d(sim_time, sim_e_field, bounds_error=False, fill_value=0.0)

    def ect_waveform(self, stimulus_duration: float, pulse_width: float, pulse_time: float, time_step: float = 0.001) -> Callable[[float], float]:
        time = []
        intensity = []
        t = 0
        while t <= stimulus_duration:
            time.append(t)
            intensity.append(1.0 if pulse_time <= t < pulse_time + pulse_width else 0.0)
            t += time_step

        self.time = np.array(time)
        self.e_field_magnitude = np.array(intensity)
        return interp1d(self.time, self.e_field_magnitude, bounds_error=False, fill_value=0.0)


def select_waveform(stim_type: str, pulse_width: float) -> Tuple[Callable[[float], float], float, float]:
    if stim_type == "TMS":
        waveform = Waveform(WaveformType.BIPHASIC)
        dt = 0.001
        delay = 2.0
        tstop = 50.0
        wf_callable = waveform.load_waveform(dt, delay, tstop)
    elif stim_type == "MST":
        waveform = Waveform(WaveformType.MST_BIPHASIC)
        dt = 0.001
        delay = 2.0
        tstop = 50.0
        wf_callable = waveform.load_waveform(dt, delay, tstop)
    elif stim_type == "ECT":
        waveform = Waveform(WaveformType.RECTANGULAR)
        dt = 0.001
        tstop = 15.0
        pulse_time = 2.0
        wf_callable = waveform.ect_waveform(
            stimulus_duration=tstop,
            pulse_width=pulse_width,
            pulse_time=pulse_time,
            time_step=dt
        )
    else:
        raise ValueError(f"Unknown stim_type: {stim_type}")

    return wf_callable, dt, tstop
