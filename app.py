"""
Biomechanical Movement Tracker - Streamlit Cloud Compatible
All-in-one version without external module dependencies
"""
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tempfile
import os

# ============= BIOMECHANICAL CALCULATOR =============
class BiomechanicalCalculator:
    """Biomechanically optimal formulas for energy expenditure"""
    
    def __init__(self, weight_kg=70, height_cm=170, age=30, sex='male'):
        self.weight = weight_kg
        self.height = height_cm
        self.age = age
        self.sex = sex
        self.lean_mass = self.estimate_lean_mass()
        
    def estimate_lean_mass(self):
        """Boer formula (1984) for lean body mass"""
        if self.sex == 'male':
            return (0.407 * self.weight) + (0.267 * self.height) - 19.2
        else:
            return (0.252 * self.weight) + (0.473 * self.height) - 48.3
    
    def calculate_energy_expenditure(self, velocity_data, acceleration_data, duration_s):
        """Calculate energy in Joules using biomechanical work-energy theorem"""
        if len(velocity_data) == 0:
            return 0
            
        # Kinetic energy changes
        ke_changes = 0.5 * self.weight * np.sum(np.diff(velocity_data**2)) if len(velocity_data) > 1 else 0
        
        # Work against gravity
        if isinstance(acceleration_data, np.ndarray) and len(acceleration_data.shape) > 1:
            vertical_work = self.weight * 9.81 * np.sum(np.abs(np.diff(acceleration_data[:, 1])))
        else:
            vertical_work = self.weight * 9.81 * np.sum(np.abs(acceleration_data)) * 0.1
        
        # Internal work (10% of total)
        internal_work = 0.1 * self.weight * np.sum(np.abs(acceleration_data))
        
        # Total mechanical work
        total_work = abs(ke_changes) + vertical_work + internal_work
        
        # Metabolic efficiency 23% (Cavagna & Kaneko, 1977)
        metabolic_energy = total_work / 0.23
        
        # Add BMR component
        bmr_joules_per_second = (500 + 22 * self.lean_mass) * 4.184 / 86400
        bmr_component = bmr_joules_per_second * duration_s
        
        return metabolic_energy + bmr_component
    
    def estimate_protein_needs(self, energy_joules, activity_intensity):
        """Protein requirements based on ISSN (2017) & Moore et al. (2015)"""
        energy_kcal = energy_joules / 4184
        
        # Protein factor based on intensity
        if activity_intensity < 3:
            protein_factor = 0.8
        elif activity_intensity < 6:
            protein_factor = 1.2
        elif activity_intensity < 9:
            protein_factor = 1.6
        else:
            protein_factor = 2.0
        
        session_protein = (protein_factor * self.weight * energy_kcal) / 2000
        
        # Minimum 20g for MPS (Moore et al., 2015)
        return max(session_protein, 20, min(session_protein, 40))

# ============= VIDEO PROCESSOR =============
class VideoProcessor:
    """Process video using optical flow for movement detection"""
    
    def __init__(self):
        self.flow_params = dict(
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        self.timestamps = []
        
    def process_video(self, video_path, progress_bar=None):
        """Process video using Farneback optical flow"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps == 0 or fps > 120:
                fps = 30
                
            ret, frame1 = cap.read()
            if not ret:
                return None, None, None
                
            # Resize for performance
            h, w = frame1.shape[:2]
            if w > 640:
                scale = 640 / w
                new_dim = (int(w * scale), int(h * scale))
                frame1 = cv2.resize(frame1, new_dim)
                
            prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            flow_magnitudes = []
            frame_count = 0
            
            while True:
                ret, frame2 = cap.read()
                if not ret:
                    break
                    
                if w > 640:
                    frame2 = cv2.resize(frame2, new_dim)
                    
                next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prvs, next_gray, None, **self.flow_params)
                
                magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                
                # Filter outliers
                mag_flat = magnitude.flatten()
                threshold = np.percentile(mag_flat, 95)
                mag_filtered = mag_flat[mag_flat < threshold]
                avg_mag = np.mean(mag_filtered) if len(mag_filtered) > 0 else 0
                
                flow_magnitudes.append(avg_mag)
                self.timestamps.append(frame_count / fps)
                
                prvs = next_gray
                frame_count += 1
                
                if progress_bar:
                    progress_bar.progress(min(frame_count / total_frames, 1.0))
                    
            cap.release()
            
            if len(flow_magnitudes) > 0:
                return self.analyze_motion(flow_magnitudes, fps)
            return None, None, None
            
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            return None, None, None
    
    def analyze_motion(self, flow_magnitudes, fps):
        """Convert optical flow to movement metrics"""
        if len(flow_magnitudes) < 2:
            return None, None, None
        
        flow_array = np.array(flow_magnitudes)
        flow_array = flow_array[flow_array > 0]
        
        if len(flow_array) < 2:
            return None, None, None
            
        # Scale to velocity (pixels to meters)
        pixel_to_meter = 1.7 / 300  # Assuming 1.7m person = 300 pixels
        velocity = flow_array * pixel_to_meter * fps
        
        # Smooth signal
        if len(velocity) > 5:
            window = min(5, len(velocity) // 4)
            velocity = np.convolve(velocity, np.ones(window)/window, mode='same')
        
        velocity = np.clip(velocity, 0, 10)  # Cap at 10 m/s
        
        # Calculate acceleration
        dt = 1 / fps
        if len(velocity) > 1:
            accel = np.gradient(velocity) / dt
            acceleration = np.column_stack([accel * 0.7, np.abs(accel) * 0.3])
        else:
            acceleration = np.zeros((len(velocity), 2))
        
        # Estimate METs
        mean_v = np.mean(velocity)
        if mean_v < 0.5:
            intensity = 2.0
        elif mean_v < 1.0:
            intensity = 4.0
        elif mean_v < 2.0:
            intensity = 7.0
        elif mean_v < 3.0:
            intensity = 10.0
        else:
            intensity = 12.0
            
        return velocity, acceleration, intensity

# ============= VISUALIZATION =============
def create_velocity_plot(timestamps, velocity_data):
    """Create velocity profile plot"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=velocity_data,
        mode='lines', name='Velocity',
        line=dict(color='#3498db', width=2),
        fill='tonexty', fillcolor='rgba(52, 152, 219, 0.2)'
    ))
    
    fig.update_layout(
        title="Velocity Profile",
        xaxis_title="Time (s)",
        yaxis_title="Velocity (m/s)",
        height=350
    )
    return fig

def create_energy_plot(timestamps, energy_data):
    """Create energy expenditure plot"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=energy_data,
        mode='lines', name='Cumulative Energy',
        line=dict(color='#27ae60', width=3),
        fill='tozeroy', fillcolor='rgba(39, 174, 96, 0.2)'
    ))
    
    fig.update_layout(
        title="Energy Expenditure Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Energy (J)",
        height=350
    )
    return fig

# ============= MAIN APP =============
def main():
    st.set_page_config(
        page_title="Biomechanical Movement Tracker",
        layout="wide",
        page_icon="🏃"
    )
    
    st.title("🏃 Biomechanical Movement Analysis")
    st.markdown("""
    Calculate energy expenditure (Joules) and protein requirements from video analysis.
    Based on peer-reviewed biomechanical formulas.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("User Parameters")
        weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)
        height = st.number_input("Height (cm)", 100.0, 250.0, 170.0)
        age = st.number_input("Age", 10, 100, 30)
        sex = st.selectbox("Sex", ["male", "female"])
        
        st.markdown("---")
        st.caption("**References:**")
        st.caption("• Cavagna & Kaneko (1977)")
        st.caption("• Cunningham (1991)")
        st.caption("• ISSN (2017)")
        st.caption("• Moore et al. (2015)")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Video (MP4, AVI, MOV)",
        type=['mp4', 'avi', 'mov']
    )
    
    if uploaded_file:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Analysis")
            progress = st.progress(0)
            
            # Process
            processor = VideoProcessor()
            calculator = BiomechanicalCalculator(weight, height, age, sex)
            
            with st.spinner("Processing..."):
                velocity, acceleration, intensity = processor.process_video(temp_path, progress)
            
            if velocity is not None and len(velocity) > 0:
                duration = len(velocity) / 30  # Approximate
                
                # Calculate
                energy_j = calculator.calculate_energy_expenditure(velocity, acceleration, duration)
                protein_g = calculator.estimate_protein_needs(energy_j, intensity)
                
                st.success("✅ Complete!")
                
                # Metrics
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Energy", f"{energy_j:.0f} J", f"{energy_j/4184:.1f} kcal")
                with c2:
                    st.metric("Protein", f"{protein_g:.1f} g", f"{protein_g/weight:.2f} g/kg")
                with c3:
                    st.metric("Intensity", f"{intensity:.1f} METs")
            else:
                st.error("Could not analyze video")
        
        with col2:
            if 'velocity' in locals() and velocity is not None:
                st.subheader("📈 Plots")
                
                timestamps = np.linspace(0, len(velocity)/30, len(velocity))
                
                # Velocity plot
                st.plotly_chart(create_velocity_plot(timestamps, velocity), use_container_width=True)
                
                # Energy plot
                energy_cumulative = np.cumsum(np.abs(velocity)) * weight * 9.81 / 0.23
                st.plotly_chart(create_energy_plot(timestamps, energy_cumulative), use_container_width=True)
        
        # Cleanup
        try:
            os.unlink(temp_path)
        except:
            pass

if __name__ == "__main__":
    main()
