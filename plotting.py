import matplotlib.pyplot as plt
import pandas as pd
from agents.base_agent import AgentResult

def plot_hr_trend(data: dict):
    """Plots a heart rate trend."""
    df = pd.DataFrame(data['samples'])
    df['ts'] = pd.to_datetime(df['ts'], unit='s')
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['ts'], df['value'], marker='o', linestyle='-', color='b')
    ax.set_title(f"Heart Rate Trend for Patient {data.get('patient_id', 'N/A')}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Heart Rate (bpm)")
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_ecg_waveform(data: dict):
    """Plots an ECG waveform."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(data['t'], data['signal'], color='r')
    ax.set_title(f"ECG Waveform (Lead {data.get('lead', 'N/A')}) for Patient {data.get('patient_id', 'N/A')}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (mV)")
    ax.grid(True)
    plt.tight_layout()
    return fig

def plot_ews_trend(data: dict):
    """Plots an Early Warning Score trend."""
    df = pd.DataFrame(data['samples'])
    df['ts'] = pd.to_datetime(df['ts'], unit='s')
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['ts'], df['value'], marker='^', linestyle='--', color='g')
    ax.set_title(f"EWS Trend for Patient {data.get('patient_id', 'N/A')}")
    ax.set_xlabel("Time")
    ax.set_ylabel("EWS Score")
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_agent_result(result: AgentResult):
    """
    Master plotting function that routes to the correct plotter
    based on the result's plot_type.
    """
    if not result.plot_data or not result.plot_type:
        return None

    plot_functions = {
        "hr_trend": plot_hr_trend,
        "ecg_waveform": plot_ecg_waveform,
        "ews_trend": plot_ews_trend,
    }

    plot_function = plot_functions.get(result.plot_type)
    
    if plot_function:
        return plot_function(result.plot_data)
        
    return None

