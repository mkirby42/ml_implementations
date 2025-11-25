import numpy as np

def generate_signal(sample_rate, frequency, amplitude, noise_level, phase_offset, duration=1.0):
    """
    Generate a sinusoidal signal with specified parameters. Can generate chirp signals with varying frequency.
    
    Parameters:
    -----------
    sample_rate : float
        Number of samples per second (Hz)
    frequency : float or tuple of (float, float)
        Frequency of the sine wave (Hz). Can be:
        - A single float for a stationary frequency signal
        - A tuple (start_freq, end_freq) for a chirp signal with linearly varying frequency
    amplitude : float
        Amplitude of the sine wave
    noise_level : float
        Standard deviation of Gaussian noise to add
    phase_offset : float
        Phase offset in radians
    duration : float, optional
        Duration of the signal in seconds (default: 1.0)
    
    Returns:
    --------
    t : numpy.ndarray
        Time array
    signal : numpy.ndarray
        Generated signal with noise
    """
    # Generate time array
    t = np.arange(0, duration, 1/sample_rate)
    
    # Parse frequency parameter
    if isinstance(frequency, (tuple, list)):
        start_frequency = frequency[0]
        end_frequency = frequency[1]
    else:
        start_frequency = frequency
        end_frequency = frequency
    
    # Generate clean signal
    if start_frequency == end_frequency:
        # Stationary frequency signal
        clean_signal = amplitude * np.sin(2 * np.pi * start_frequency * t + phase_offset)
    else:
        # Chirp signal with linearly varying frequency
        # Instantaneous frequency: f(t) = start_frequency + (end_frequency - start_frequency) * t / duration
        # Phase: integral of 2*pi*f(t) dt = 2*pi * (start_frequency*t + (end_frequency - start_frequency)*t^2 / (2*duration))
        instantaneous_phase = 2 * np.pi * (start_frequency * t + (end_frequency - start_frequency) * t**2 / (2 * duration))
        clean_signal = amplitude * np.sin(instantaneous_phase + phase_offset)
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, len(t))
    signal = clean_signal + noise
    
    return t, signal
