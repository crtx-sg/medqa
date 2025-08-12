import matplotlib.pyplot as plt
import pandas as pd
from agents.base_agent import AgentResult
import datetime

def display_single_patient_dashboard(data: dict, st):
    """Renders a complex dashboard for a single patient in the Streamlit app."""
    st.subheader(f"Dashboard for {data.get('patient_name', 'N/A')} (ID: {data.get('patient_id', 'N/A')})")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Ward", data.get('ward_name', 'N/A'))
        st.metric("Location", f"{data.get('room', 'N/A')}-{data.get('bed', 'N/A')}")
    with col2:
        st.metric("Attending Physician", data.get('doctor_name', 'N/A'))
        st.metric("EWS Score", data.get('vitals', {}).get('ews_score', 'N/A'))
    with col3:
        st.write("**Medications:**")
        st.json(data.get('medications', {}))
        st.write("**History:**")
        st.json(data.get('history', []))

    st.divider()
    st.subheader("Latest Vitals")
    vitals = data.get('vitals', {})
    v1, v2, v3, v4, v5 = st.columns(5)
    v1.metric("Heart Rate (bpm)", vitals.get('hr', 'N/A'))
    v2.metric("Resp. Rate", vitals.get('rr', 'N/A'))
    v3.metric("Blood Pressure", vitals.get('bp', 'N/A'))
    v4.metric("SpO2 (%)", vitals.get('spo2', 'N/A'))
    v5.metric("Temperature (°F)", vitals.get('temperature', 'N/A'))

    st.divider()
    st.subheader("Latest ECG Strip")
    ecg_data = data.get('ecg', [])
    if ecg_data:
        fig, ax = plt.subplots(figsize=(12, 3))
        fs = 250
        time_axis = [i / fs for i in range(len(ecg_data))]
        ax.plot(time_axis, ecg_data, color='r')
        ax.set_title("ECG Waveform (Lead II)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (mV)")
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.write("No ECG data available.")

def display_all_patients_table(data: dict):
    """Prepares the all-patient snapshot for display in a table."""
    if not data or 'patients' not in data:
        return pd.DataFrame()
    
    table_data = {k: v for k, v in data.items() if k != 'vitals'}
    df = pd.DataFrame(table_data)
    
    if 'vitals' in data and len(data['vitals']) == len(df):
        df['ews_score'] = [v.get('ews_score', 'N/A') for v in data['vitals']]
    
    display_columns = ['patients', 'patient_id', 'ward_name', 'room', 'bed', 'doctor_name', 'ews_score']
    df = df[[col for col in display_columns if col in df.columns]]
    df = df.rename(columns={"patients": "Name", "patient_id": "ID", "ward_name": "Ward", "room": "Room", "bed": "Bed", "doctor_name": "Doctor", "ews_score": "EWS"})
    return df

def display_vitals_trend_table(data: dict):
    """Prepares patient vitals trend data for display in a table."""
    if 'vitals' not in data:
        return pd.DataFrame()
        
    df = pd.DataFrame(data['vitals'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
    
    display_columns = ['timestamp', 'hr', 'rr', 'bp', 'temperature', 'spo2', 'ews_score']
    df = df[[col for col in display_columns if col in df.columns]]
    return df

def plot_patient_vitals_trend(data: dict):
    """Plots the vitals trend for a single patient with multiple subplots."""
    if 'vitals' not in data or not data['vitals']:
        return None

    df = pd.DataFrame(data['vitals'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    if 'bp' in df.columns:
        bp_split = df['bp'].str.split('/', expand=True)
        df['bp_sys'] = pd.to_numeric(bp_split[0])
        df['bp_dia'] = pd.to_numeric(bp_split[1])

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f"Vitals Trend for {data.get('patients', 'N/A')}", fontsize=16)
    axes = axes.flatten()

    vitals_to_plot = {
        'hr': {'ax': axes[0], 'label': 'Heart Rate (bpm)', 'color': 'red'},
        'rr': {'ax': axes[1], 'label': 'Respiratory Rate', 'color': 'blue'},
        'spo2': {'ax': axes[2], 'label': 'SpO2 (%)', 'color': 'green'},
        'temperature': {'ax': axes[3], 'label': 'Temperature (°F)', 'color': 'orange'},
        'ews_score': {'ax': axes[4], 'label': 'EWS Score', 'color': 'purple'},
    }

    for key, props in vitals_to_plot.items():
        if key in df.columns:
            props['ax'].plot(df['timestamp'], df[key], marker='.', linestyle='-', color=props['color'])
            props['ax'].set_title(props['label'])
            props['ax'].grid(True, alpha=0.5)
            props['ax'].tick_params(axis='x', rotation=45)

    if 'bp_sys' in df.columns and 'bp_dia' in df.columns:
        axes[5].plot(df['timestamp'], df['bp_sys'], marker='.', linestyle='-', color='black', label='Systolic')
        axes[5].plot(df['timestamp'], df['bp_dia'], marker='.', linestyle='-', color='grey', label='Diastolic')
        axes[5].set_title('Blood Pressure (mmHg)')
        axes[5].grid(True, alpha=0.5)
        axes[5].tick_params(axis='x', rotation=45)
        axes[5].legend()
    else:
        fig.delaxes(axes[5])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
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

def plot_agent_result(result: AgentResult, st):
    """
    Master display function that routes to the correct renderer.
    This function calls the appropriate Streamlit methods to display content.
    """
    if not result.plot_data or not result.plot_type:
        return

    if result.plot_type == "single_patient_dashboard":
        display_single_patient_dashboard(result.plot_data, st)
        return

    if result.plot_type == "all_patients_table":
        df = display_all_patients_table(result.plot_data)
        st.dataframe(df)
        return
        
    if result.plot_type == "vitals_table":
        df = display_vitals_trend_table(result.plot_data)
        st.dataframe(df)
        return

    fig = None
    if result.plot_type == "ews_trend":
        fig = plot_ews_trend(result.plot_data)
    elif result.plot_type == "vitals_trend":
        fig = plot_patient_vitals_trend(result.plot_data)
    
    if fig:
        st.pyplot(fig)

