import os
import sys
import pickle
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.data_generation import generate_composite_signal, SignalConfig


def generate_dataset(n_sessions=10, sample_rate=100):
    """
    Generate synthetic composite signals for training.
    
    Args:
        n_sessions: Number of signal sessions to generate
        sample_rate: Sampling rate in Hz
    
    Returns:
        List of generated signals
    """
    signals = []
    
    for session_idx in range(n_sessions):
        # Generate random session parameters
        duration_seconds = np.random.normal(45 * 60, 10 * 60)
        duration_seconds = np.clip(duration_seconds, 15 * 60, 90 * 60)
        
        # Dominant signal parameters
        dominant_signal_starting_frequency = np.random.normal(0.2, 0.08)
        dominant_signal_ending_frequency = np.random.normal(0.7, 0.08)
        dominant_signal_amplitude = np.random.normal(200, 50)
        dominant_signal_noise_level = np.random.normal(5, 1)
        dominant_signal_phase_offset = np.random.normal(0, 0.1)
        
        # Secondary signal parameters
        secondary_signal_starting_frequency = np.random.normal(1.1, 0.08)
        secondary_signal_ending_frequency = np.random.normal(2.5, 0.08)
        secondary_signal_amplitude = np.random.normal(10, 2)
        secondary_signal_noise_level = np.random.normal(5, 1)
        secondary_signal_phase_offset = np.random.normal(0, 0.1)
        
        # Create signal configurations
        signal_configs = [
            SignalConfig(
                frequency=[dominant_signal_starting_frequency, dominant_signal_ending_frequency], 
                amplitude=dominant_signal_amplitude, 
                noise_level=dominant_signal_noise_level, 
                phase_offset=dominant_signal_phase_offset
            ),
            SignalConfig(
                frequency=[secondary_signal_starting_frequency, secondary_signal_ending_frequency], 
                amplitude=secondary_signal_amplitude, 
                noise_level=secondary_signal_noise_level, 
                phase_offset=secondary_signal_phase_offset
            ),
        ]
        
        # Generate composite signal
        t, sensor_signal_1 = generate_composite_signal(
            duration_seconds=duration_seconds, 
            sample_rate=sample_rate, 
            signal_configs=signal_configs
        )
        signals.append(sensor_signal_1)
        
        print(f"Generated session {session_idx + 1}/{n_sessions}")
    
    return signals


def save_data(signals, output_path="data/signals.pkl"):
    """
    Save signals to disk.
    
    Args:
        signals: List of signals to save
        output_path: Path to save the generated signals
    """
    # Save signals to disk
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(signals, f)
    
    print(f"Successfully saved {len(signals)} signals to {output_path}")


if __name__ == "__main__":
    signals = generate_dataset()
    save_data(signals)