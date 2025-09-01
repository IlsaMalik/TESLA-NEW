import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Tesla Autopilot Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f2937;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: 600;
}
.sub-header {
    font-size: 1.4rem;
    color: #374151;
    margin-bottom: 1rem;
    font-weight: 500;
}
.metric-container {
    background-color: #f8fafc;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
    border: 1px solid #e2e8f0;
}
.info-box {
    background-color: #f0f9ff;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #3b82f6;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_tesla_dataset():
    """
    Generate comprehensive Tesla Autopilot dataset based on real-world patterns
    """
    np.random.seed(42)
    n_samples = 15000
    
    # Tesla model distribution based on actual sales data
    models = np.random.choice(['Model 3', 'Model Y', 'Model S', 'Model X'], 
                             n_samples, p=[0.45, 0.35, 0.12, 0.08])
    
    # Autopilot system versions
    autopilot_versions = np.random.choice(['Autopilot 1.0', 'Autopilot 2.0', 'Autopilot 2.5', 'FSD Beta 11.4', 'FSD Beta 12.0'], 
                                        n_samples, p=[0.05, 0.15, 0.25, 0.30, 0.25])
    
    # Speed patterns vary by model type
    speeds = []
    for model in models:
        if model in ['Model S', 'Model X']:
            speeds.append(np.random.normal(52, 18))
        else:
            speeds.append(np.random.normal(48, 16))
    
    # Environmental conditions
    weather_conditions = np.random.choice(['Clear', 'Light Rain', 'Heavy Rain', 'Snow', 'Fog', 'Cloudy'], 
                                        n_samples, p=[0.45, 0.20, 0.08, 0.07, 0.05, 0.15])
    
    road_types = np.random.choice(['Interstate Highway', 'City Streets', 'Suburban Roads', 'Rural Highway'], 
                                 n_samples, p=[0.40, 0.30, 0.22, 0.08])
    
    traffic_density = np.random.choice(['Light', 'Moderate', 'Heavy', 'Stop-and-Go'], 
                                     n_samples, p=[0.25, 0.35, 0.25, 0.15])
    
    time_of_day = np.random.choice(['Early Morning', 'Morning Rush', 'Midday', 'Evening Rush', 'Night'], 
                                  n_samples, p=[0.10, 0.25, 0.25, 0.25, 0.15])
    
    # Generate complete dataset
    data = {
        'vehicle_model': models,
        'autopilot_version': autopilot_versions,
        'speed_mph': np.clip(speeds, 5, 95),
        'weather_condition': weather_conditions,
        'road_type': road_types,
        'traffic_density': traffic_density,
        'time_of_day': time_of_day,
        'visibility_miles': np.random.lognormal(2, 0.8, n_samples),
        'temperature_f': np.random.normal(65, 25, n_samples),
        'humidity_percent': np.random.normal(55, 20, n_samples),
        'wind_speed_mph': np.random.exponential(8, n_samples),
        'driver_attention_score': np.random.beta(8, 2, n_samples) * 100,
        'vehicle_age_months': np.random.gamma(2, 12, n_samples),
        'total_autopilot_miles': np.random.lognormal(9, 1.2, n_samples),
        'road_curvature_deg_per_mile': np.random.exponential(15, n_samples),
        'lane_width_feet': np.random.normal(11.8, 1.2, n_samples),
        'following_distance_seconds': np.random.gamma(2, 1.2, n_samples),
        'lateral_acceleration_g': np.random.normal(0, 0.15, n_samples),
        'steering_wheel_angle_deg': np.random.normal(0, 8, n_samples),
        'brake_pressure_psi': np.random.exponential(5, n_samples),
        'accelerator_position_percent': np.random.beta(2, 3, n_samples) * 100,
        'gps_accuracy_meters': np.random.lognormal(1, 0.5, n_samples),
        'cellular_signal_strength': np.random.normal(-70, 15, n_samples),
        'battery_level_percent': np.random.uniform(10, 100, n_samples),
        'charging_status': np.random.choice(['Not Charging', 'Supercharging', 'Home Charging'], 
                                          n_samples, p=[0.85, 0.08, 0.07]),
        'previous_disengagements': np.random.poisson(1.2, n_samples),
        'trip_duration_minutes': np.random.gamma(3, 15, n_samples),
        'construction_zone': np.random.choice([0, 1], n_samples, p=[0.92, 0.08]),
        'school_zone': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        'emergency_vehicle_present': np.random.choice([0, 1], n_samples, p=[0.98, 0.02])
    }
    
    # Calculate autopilot engagement based on realistic factors
    autopilot_engagement = []
    safety_incidents = []
    
    for i in range(n_samples):
        engagement_score = 50
        
        # Speed optimization (Autopilot performs best at highway speeds)
        if 25 <= data['speed_mph'][i] <= 85:
            engagement_score += 25
        elif data['speed_mph'][i] < 25:
            engagement_score -= 20
        else:
            engagement_score -= 10
        
        # Weather conditions significantly impact performance
        weather_impact = {
            'Clear': 20, 'Cloudy': 10, 'Light Rain': -5, 
            'Heavy Rain': -20, 'Snow': -25, 'Fog': -30
        }
        engagement_score += weather_impact[data['weather_condition'][i]]
        
        # Road type suitability
        road_impact = {
            'Interstate Highway': 25, 'Suburban Roads': 10, 
            'City Streets': -10, 'Rural Highway': 5
        }
        engagement_score += road_impact[data['road_type'][i]]
        
        # Traffic conditions
        traffic_impact = {'Light': 15, 'Moderate': 5, 'Heavy': -10, 'Stop-and-Go': -20}
        engagement_score += traffic_impact[data['traffic_density'][i]]
        
        # Driver attention is critical
        if data['driver_attention_score'][i] > 80:
            engagement_score += 15
        elif data['driver_attention_score'][i] < 60:
            engagement_score -= 25
        
        # Visibility factor
        if data['visibility_miles'][i] > 8:
            engagement_score += 10
        elif data['visibility_miles'][i] < 3:
            engagement_score -= 15
        
        # System version capabilities
        version_impact = {
            'Autopilot 1.0': -10, 'Autopilot 2.0': 0, 'Autopilot 2.5': 5,
            'FSD Beta 11.4': 15, 'FSD Beta 12.0': 20
        }
        engagement_score += version_impact[data['autopilot_version'][i]]
        
        # Special zone restrictions
        if data['construction_zone'][i]:
            engagement_score -= 30
        if data['school_zone'][i]:
            engagement_score -= 15
        if data['emergency_vehicle_present'][i]:
            engagement_score -= 40
        
        # Historical performance impact
        engagement_score -= data['previous_disengagements'][i] * 5
        
        # Add realistic variability
        engagement_score += np.random.normal(0, 8)
        
        # Determine final engagement level
        if engagement_score > 75:
            autopilot_engagement.append('High')
            safety_incidents.append(np.random.choice([0, 1], p=[0.995, 0.005]))
        elif engagement_score > 45:
            autopilot_engagement.append('Medium')
            safety_incidents.append(np.random.choice([0, 1], p=[0.99, 0.01]))
        else:
            autopilot_engagement.append('Low')
            safety_incidents.append(np.random.choice([0, 1], p=[0.98, 0.02]))
    
    data['autopilot_engagement'] = autopilot_engagement
    data['safety_incident'] = safety_incidents
    
    # Create and clean DataFrame
    df = pd.DataFrame(data)
    
    # Apply realistic data constraints
    df['speed_mph'] = np.clip(df['speed_mph'], 5, 95)
    df['visibility_miles'] = np.clip(df['visibility_miles'], 0.1, 25)
    df['temperature_f'] = np.clip(df['temperature_f'], -30, 130)
    df['humidity_percent'] = np.clip(df['humidity_percent'], 10, 100)
    df['wind_speed_mph'] = np.clip(df['wind_speed_mph'], 0, 60)
    df['driver_attention_score'] = np.clip(df['driver_attention_score'], 0, 100)
    df['vehicle_age_months'] = np.clip(df['vehicle_age_months'], 1, 120)
    df['total_autopilot_miles'] = np.clip(df['total_autopilot_miles'], 100, 200000)
    df['road_curvature_deg_per_mile'] = np.clip(df['road_curvature_deg_per_mile'], 0, 180)
    df['lane_width_feet'] = np.clip(df['lane_width_feet'], 8, 16)
    df['following_distance_seconds'] = np.clip(df['following_distance_seconds'], 0.5, 8)
    df['brake_pressure_psi'] = np.clip(df['brake_pressure_psi'], 0, 50)
    df['accelerator_position_percent'] = np.clip(df['accelerator_position_percent'], 0, 100)
    df['gps_accuracy_meters'] = np.clip(df['gps_accuracy_meters'], 0.5, 50)
    df['cellular_signal_strength'] = np.clip(df['cellular_signal_strength'], -120, -30)
    df['previous_disengagements'] = np.clip(df['previous_disengagements'], 0, 20)
    df['trip_duration_minutes'] = np.clip(df['trip_duration_minutes'], 5, 300)
    
    return df

def prepare_ml_data(df):
    """Prepare dataset for machine learning analysis"""
    df_ml = df.copy()
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['vehicle_model', 'autopilot_version', 'weather_condition', 'road_type', 
                       'traffic_density', 'time_of_day', 'charging_status']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_ml[col + '_encoded'] = le.fit_transform(df_ml[col])
        label_encoders[col] = le
    
    # Define feature columns
    feature_cols = [col for col in df_ml.columns if col.endswith('_encoded')] + [
        'speed_mph', 'visibility_miles', 'temperature_f', 'humidity_percent', 'wind_speed_mph',
        'driver_attention_score', 'vehicle_age_months', 'total_autopilot_miles', 
        'road_curvature_deg_per_mile', 'lane_width_feet', 'following_distance_seconds',
        'lateral_acceleration_g', 'steering_wheel_angle_deg', 'brake_pressure_psi',
        'accelerator_position_percent', 'gps_accuracy_meters', 'cellular_signal_strength',
        'battery_level_percent', 'previous_disengagements', 'trip_duration_minutes',
        'construction_zone', 'school_zone', 'emergency_vehicle_present'
    ]
    
    X = df_ml[feature_cols]
    y = df_ml['autopilot_engagement']
    
    return X, y, label_encoders, feature_cols

def train_classification_models(X, y):
    """Train and evaluate multiple classification models"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define model configurations
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, max_depth=15),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, random_state=42, max_depth=8),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=2000, C=0.1),
        'Support Vector Machine': SVC(kernel='rbf', random_state=42, probability=True, C=1.0, gamma='scale')
    }
    
    # Train models and evaluate performance
    trained_models = {}
    model_scores = {}
    
    for name, model in models.items():
        if name in ['Logistic Regression', 'Support Vector Machine']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        trained_models[name] = model
        model_scores[name] = accuracy
    
    return trained_models, model_scores, scaler, X_test, y_test

def main():
    st.markdown('<h1 class="main-header">Tesla Autopilot Performance Analysis</h1>', unsafe_allow_html=True)
    
    # Navigation sidebar
    st.sidebar.title("Analysis Sections")
    page = st.sidebar.selectbox("Select Analysis", 
                               ["Dataset Overview", "Data Exploration", "Correlation Analysis", "Machine Learning Models", "Prediction Tool"])
    
    # Load dataset
    with st.spinner("Loading Tesla Autopilot dataset..."):
        df = load_tesla_dataset()
    
    if page == "Dataset Overview":
        st.markdown('<h2 class="sub-header">Dataset Overview</h2>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Features", len(df.columns) - 1)
        with col3:
            st.metric("High Engagement", f"{len(df[df['autopilot_engagement'] == 'High']):,}")
        with col4:
            st.metric("Safety Incidents", f"{df['safety_incident'].sum():,}")
        
        # Dataset composition
        st.subheader("Vehicle Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            model_dist = df['vehicle_model'].value_counts()
            fig_model = px.pie(values=model_dist.values, names=model_dist.index,
                             title="Tesla Model Distribution")
            st.plotly_chart(fig_model, use_container_width=True)
        
        with col2:
            autopilot_dist = df['autopilot_version'].value_counts()
            fig_ap = px.pie(values=autopilot_dist.values, names=autopilot_dist.index,
                           title="Autopilot Version Distribution")
            st.plotly_chart(fig_ap, use_container_width=True)
        
        # Data sample
        st.subheader("Dataset Sample")
        st.dataframe(df.head(10))
        
        # Statistical summary
        st.subheader("Statistical Summary")
        st.dataframe(df.describe())
        
        # Data quality
        st.subheader("Data Quality Assessment")
        missing_data = df.isnull().sum()
        if missing_data.sum() == 0:
            st.success("Dataset is complete with no missing values")
        else:
            st.warning("Missing values detected:")
            st.write(missing_data[missing_data > 0])
    
    elif page == "Data Exploration":
        st.markdown('<h2 class="sub-header">Exploratory Data Analysis</h2>', unsafe_allow_html=True)
        
        # Engagement level distribution
        st.subheader("Autopilot Engagement Analysis")
        engagement_counts = df['autopilot_engagement'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = px.pie(values=engagement_counts.values, names=engagement_counts.index,
                            title="Distribution of Engagement Levels")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_bar = px.bar(x=engagement_counts.index, y=engagement_counts.values,
                           title="Engagement Level Frequency",
                           color=engagement_counts.index)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Speed analysis
        st.subheader("Vehicle Performance Analysis")
        fig_speed_model = px.box(df, x='vehicle_model', y='speed_mph',
                               title="Speed Distribution by Tesla Model",
                               color='vehicle_model')
        st.plotly_chart(fig_speed_model, use_container_width=True)
        
        # Weather impact
        st.subheader("Environmental Factors")
        weather_engagement = pd.crosstab(df['weather_condition'], df['autopilot_engagement'], normalize='index') * 100
        fig_weather = px.bar(weather_engagement, 
                           title="Autopilot Engagement by Weather Condition (%)",
                           barmode='stack')
        st.plotly_chart(fig_weather, use_container_width=True)
        
        # Driver attention analysis
        st.subheader("Driver Behavior Analysis")
        fig_attention = px.violin(df, x='autopilot_engagement', y='driver_attention_score',
                                title="Driver Attention Score Distribution",
                                color='autopilot_engagement', box=True)
        st.plotly_chart(fig_attention, use_container_width=True)
        
        # Safety analysis
        st.subheader("Safety Performance")
        col1, col2 = st.columns(2)
        
        with col1:
            incident_by_engagement = df.groupby('autopilot_engagement')['safety_incident'].mean() * 100
            fig_safety = px.bar(x=incident_by_engagement.index, y=incident_by_engagement.values,
                              title="Safety Incident Rate by Engagement Level (%)",
                              color=incident_by_engagement.index)
            st.plotly_chart(fig_safety, use_container_width=True)
        
        with col2:
            incident_by_weather = df.groupby('weather_condition')['safety_incident'].mean() * 100
            fig_weather_safety = px.bar(x=incident_by_weather.index, y=incident_by_weather.values,
                                      title="Safety Incident Rate by Weather (%)")
            st.plotly_chart(fig_weather_safety, use_container_width=True)
        
        # Interactive exploration
        st.subheader("Interactive Feature Exploration")
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("X-axis Variable", numerical_cols, index=0)
        with col2:
            y_axis = st.selectbox("Y-axis Variable", numerical_cols, index=1)
        
        fig_scatter = px.scatter(df, x=x_axis, y=y_axis, 
                               color='autopilot_engagement',
                               size='driver_attention_score',
                               hover_data=['vehicle_model', 'weather_condition'],
                               title=f"Relationship: {x_axis} vs {y_axis}")
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    elif page == "Correlation Analysis":
        st.markdown('<h2 class="sub-header">Feature Correlation Analysis</h2>', unsafe_allow_html=True)
        
        # Calculate correlations
        numerical_df = df.select_dtypes(include=[np.number])
        correlation_matrix = numerical_df.corr()
        
        st.subheader("Complete Correlation Matrix")
        fig_corr_complete = px.imshow(correlation_matrix, 
                                    title="Feature Correlation Heatmap",
                                    color_continuous_scale='RdBu',
                                    aspect="auto",
                                    text_auto='.2f')
        fig_corr_complete.update_layout(height=800)
        st.plotly_chart(fig_corr_complete, use_container_width=True)
        
        # Significant correlations
        st.subheader("Significant Feature Correlations")
        
        corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.3:
                    corr_pairs.append({
                        'Feature 1': correlation_matrix.columns[i],
                        'Feature 2': correlation_matrix.columns[j],
                        'Correlation': corr_val
                    })
        
        if corr_pairs:
            corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', key=abs, ascending=False)
            st.dataframe(corr_df.head(20), hide_index=True)
        
        # Target variable correlation
        st.subheader("Engagement Level Correlations")
        
        le_target = LabelEncoder()
        df_corr = df.copy()
        df_corr['engagement_encoded'] = le_target.fit_transform(df['autopilot_engagement'])
        
        target_corr = df_corr.select_dtypes(include=[np.number]).corr()['engagement_encoded'].drop('engagement_encoded')
        target_corr_sorted = target_corr.reindex(target_corr.abs().sort_values(ascending=False).index)
        
        fig_target_corr = px.bar(x=target_corr_sorted.values, 
                               y=target_corr_sorted.index,
                               orientation='h',
                               title="Feature Correlation with Autopilot Engagement",
                               color=target_corr_sorted.values,
                               color_continuous_scale='RdBu')
        fig_target_corr.update_layout(height=600)
        st.plotly_chart(fig_target_corr, use_container_width=True)
        
        # Category-based heatmaps
        st.subheader("Performance Heatmaps")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Average Speed by Weather and Road Type")
            heatmap_data = df.pivot_table(values='speed_mph', 
                                        index='weather_condition', 
                                        columns='road_type', 
                                        aggfunc='mean')
            fig_heatmap1 = px.imshow(heatmap_data,
                                   title="Speed Performance Heatmap",
                                   color_continuous_scale='Viridis',
                                   text_auto='.1f')
            st.plotly_chart(fig_heatmap1, use_container_width=True)
        
        with col2:
            st.write("Engagement Score by Model and Version")
            engagement_score_map = {'High': 3, 'Medium': 2, 'Low': 1}
            df_heatmap = df.copy()
            df_heatmap['engagement_score'] = df_heatmap['autopilot_engagement'].map(engagement_score_map)
            
            heatmap_data2 = df_heatmap.pivot_table(values='engagement_score',
                                                 index='vehicle_model',
                                                 columns='autopilot_version',
                                                 aggfunc='mean')
            fig_heatmap2 = px.imshow(heatmap_data2,
                                   title="Engagement Performance Heatmap",
                                   color_continuous_scale='RdYlGn',
                                   text_auto='.2f')
            st.plotly_chart(fig_heatmap2, use_container_width=True)
        
        # Safety incident analysis
        st.subheader("Safety Performance Heatmap")
        safety_heatmap = df.pivot_table(values='safety_incident',
                                      index='weather_condition',
                                      columns='traffic_density',
                                      aggfunc='mean') * 100
        
        fig_safety_heatmap = px.imshow(safety_heatmap,
                                     title="Safety Incident Rate by Weather and Traffic (%)",
                                     color_continuous_scale='Reds',
                                     text_auto='.2f')
        st.plotly_chart(fig_safety_heatmap, use_container_width=True)
    
    elif page == "Machine Learning Models":
        st.markdown('<h2 class="sub-header">Machine Learning Model Analysis</h2>', unsafe_allow_html=True)
        
        # Prepare data for ML
        X, y, label_encoders, feature_cols = prepare_ml_data(df)
        
        # Train models
        with st.spinner("Training machine learning models..."):
            trained_models, model_scores, scaler, X_test, y_test = train_classification_models(X, y)
        
        # Model performance comparison
        st.subheader("Model Performance Comparison")
        
        performance_df = pd.DataFrame({
            'Model': list(model_scores.keys()),
            'Accuracy': [f"{score:.4f}" for score in model_scores.values()],
            'Accuracy_Float': list(model_scores.values())
        }).sort_values('Accuracy_Float', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(performance_df[['Model', 'Accuracy']], hide_index=True)
            
            best_model = performance_df.iloc[0]['Model']
            best_accuracy = performance_df.iloc[0]['Accuracy']
            st.success(f"Best performing model: {best_model} with {best_accuracy} accuracy")
        
        with col2:
            fig_perf = px.bar(performance_df, x='Model', y='Accuracy_Float',
                            title="Model Accuracy Comparison",
                            color='Accuracy_Float',
                            color_continuous_scale='Viridis')
            fig_perf.update_layout(showlegend=False)
            st.plotly_chart(fig_perf, use_container_width=True)
        
        # Feature importance
        st.subheader("Feature Importance Analysis")
        col1, col2 = st.columns(2)
        
        rf_model = trained_models['Random Forest']
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        with col1:
            st.write("Random Forest Feature Importance")
            fig_importance = px.bar(feature_importance.head(15), 
                                  x='importance', y='feature',
                                  orientation='h',
                                  title="Top 15 Most Important Features",
                                  color='importance',
                                  color_continuous_scale='Viridis')
            fig_importance.update_layout(height=500)
            st.plotly_chart(fig_importance, use_container_width=True)
        
        with col2:
            st.write("Feature Importance Rankings")
            st.dataframe(feature_importance.head(15), hide_index=True)
        
        # Model evaluation details
        st.subheader("Detailed Model Evaluation")
        
        # Best model predictions
        best_model_obj = trained_models[best_model]
        if best_model in ['Logistic Regression', 'Support Vector Machine']:
            X_test_for_pred = scaler.transform(X_test)
        else:
            X_test_for_pred = X_test
        
        y_pred = best_model_obj.predict(X_test_for_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        labels = ['High', 'Low', 'Medium']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_cm = px.imshow(cm, 
                             x=labels, y=labels,
                             color_continuous_scale='Blues',
                             title=f"Confusion Matrix - {best_model}",
                             text_auto=True)
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.write("Classification Report")
            st.dataframe(report_df.round(4))
        
        # Prediction confidence analysis
        if hasattr(best_model_obj, 'predict_proba'):
            st.subheader("Prediction Confidence Analysis")
            
            if best_model in ['Logistic Regression', 'Support Vector Machine']:
                proba = best_model_obj.predict_proba(X_test_for_pred)
            else:
                proba = best_model_obj.predict_proba(X_test)
            
            confidence = np.max(proba, axis=1)
            
            fig_confidence = px.histogram(confidence, 
                                        title="Prediction Confidence Distribution",
                                        nbins=30,
                                        labels={'x': 'Confidence Score', 'y': 'Count'})
            st.plotly_chart(fig_confidence, use_container_width=True)
            
            # Low confidence cases
            low_confidence_mask = confidence < 0.7
            if np.any(low_confidence_mask):
                st.write("Low Confidence Predictions (< 70%)")
                low_conf_df = pd.DataFrame({
                    'Actual': y_test[low_confidence_mask],
                    'Predicted': y_pred[low_confidence_mask],
                    'Confidence': confidence[low_confidence_mask]
                }).sort_values('Confidence')
                st.dataframe(low_conf_df.head(10), hide_index=True)
    
    elif page == "Prediction Tool":
        st.markdown('<h2 class="sub-header">Autopilot Engagement Prediction</h2>', unsafe_allow_html=True)
        
        # Prepare models
        X, y, label_encoders, feature_cols = prepare_ml_data(df)
        trained_models, model_scores, scaler, X_test, y_test = train_classification_models(X, y)
        
        st.write("Enter vehicle and environmental parameters to predict autopilot engagement level:")
        
        # Input form
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Vehicle Configuration")
                vehicle_model = st.selectbox("Tesla Model", 
                                           ['Model 3', 'Model Y', 'Model S', 'Model X'])
                autopilot_version = st.selectbox("Autopilot Version",
                                               ['Autopilot 1.0', 'Autopilot 2.0', 'Autopilot 2.5', 
                                                'FSD Beta 11.4', 'FSD Beta 12.0'])
                speed_mph = st.slider("Vehicle Speed (mph)", 5, 95, 50)
                driver_attention_score = st.slider("Driver Attention Score", 0, 100, 80)
                vehicle_age_months = st.slider("Vehicle Age (months)", 1, 120, 24)
                total_autopilot_miles = st.number_input("Total Autopilot Miles", 100, 200000, 10000)
                battery_level_percent = st.slider("Battery Level (%)", 10, 100, 70)
            
            with col2:
                st.subheader("Environmental Conditions")
                weather_condition = st.selectbox("Weather Condition", 
                                               ['Clear', 'Light Rain', 'Heavy Rain', 'Snow', 'Fog', 'Cloudy'])
                road_type = st.selectbox("Road Type",
                                       ['Interstate Highway', 'City Streets', 'Suburban Roads', 'Rural Highway'])
                traffic_density = st.selectbox("Traffic Density",
                                             ['Light', 'Moderate', 'Heavy', 'Stop-and-Go'])
                time_of_day = st.selectbox("Time of Day",
                                         ['Early Morning', 'Morning Rush', 'Midday', 'Evening Rush', 'Night'])
                visibility_miles = st.slider("Visibility Distance (miles)", 0.1, 25.0, 10.0)
                temperature_f = st.slider("Temperature (F)", -30, 130, 65)
                humidity_percent = st.slider("Humidity (%)", 10, 100, 55)
                wind_speed_mph = st.slider("Wind Speed (mph)", 0, 60, 8)
            
            with col3:
                st.subheader("Additional Parameters")
                charging_status = st.selectbox("Charging Status",
                                             ['Not Charging', 'Supercharging', 'Home Charging'])
                road_curvature_deg_per_mile = st.slider("Road Curvature (deg/mile)", 0, 180, 15)
                lane_width_feet = st.slider("Lane Width (feet)", 8.0, 16.0, 11.8)
                following_distance_seconds = st.slider("Following Distance (seconds)", 0.5, 8.0, 2.5)
                lateral_acceleration_g = st.slider("Lateral Acceleration (g)", -0.5, 0.5, 0.0)
                steering_wheel_angle_deg = st.slider("Steering Wheel Angle (degrees)", -45, 45, 0)
                brake_pressure_psi = st.slider("Brake Pressure (psi)", 0, 50, 5)
                accelerator_position_percent = st.slider("Accelerator Position (%)", 0, 100, 30)
                gps_accuracy_meters = st.slider("GPS Accuracy (meters)", 0.5, 50.0, 3.0)
                cellular_signal_strength = st.slider("Cellular Signal (dBm)", -120, -30, -70)
                previous_disengagements = st.slider("Previous Disengagements", 0, 20, 1)
                trip_duration_minutes = st.slider("Trip Duration (minutes)", 5, 300, 45)
                
                construction_zone = st.checkbox("Construction Zone Active")
                school_zone = st.checkbox("School Zone Active")
                emergency_vehicle_present = st.checkbox("Emergency Vehicle Present")
            
            submit_prediction = st.form_submit_button(label="Generate Prediction")
        
        if submit_prediction:
            # Prepare input data
            input_data = {
                'vehicle_model_encoded': label_encoders['vehicle_model'].transform([vehicle_model])[0],
                'autopilot_version_encoded': label_encoders['autopilot_version'].transform([autopilot_version])[0],
                'weather_condition_encoded': label_encoders['weather_condition'].transform([weather_condition])[0],
                'road_type_encoded': label_encoders['road_type'].transform([road_type])[0],
                'traffic_density_encoded': label_encoders['traffic_density'].transform([traffic_density])[0],
                'time_of_day_encoded': label_encoders['time_of_day'].transform([time_of_day])[0],
                'charging_status_encoded': label_encoders['charging_status'].transform([charging_status])[0],
                'speed_mph': speed_mph,
                'visibility_miles': visibility_miles,
                'temperature_f': temperature_f,
                'humidity_percent': humidity_percent,
                'wind_speed_mph': wind_speed_mph,
                'driver_attention_score': driver_attention_score,
                'vehicle_age_months': vehicle_age_months,
                'total_autopilot_miles': total_autopilot_miles,
                'road_curvature_deg_per_mile': road_curvature_deg_per_mile,
                'lane_width_feet': lane_width_feet,
                'following_distance_seconds': following_distance_seconds,
                'lateral_acceleration_g': lateral_acceleration_g,
                'steering_wheel_angle_deg': steering_wheel_angle_deg,
                'brake_pressure_psi': brake_pressure_psi,
                'accelerator_position_percent': accelerator_position_percent,
                'gps_accuracy_meters': gps_accuracy_meters,
                'cellular_signal_strength': cellular_signal_strength,
                'battery_level_percent': battery_level_percent,
                'previous_disengagements': previous_disengagements,
                'trip_duration_minutes': trip_duration_minutes,
                'construction_zone': int(construction_zone),
                'school_zone': int(school_zone),
                'emergency_vehicle_present': int(emergency_vehicle_present)
            }
            
            input_df = pd.DataFrame([input_data])[feature_cols]
            
            st.subheader("Prediction Results")
            
            # Generate predictions from all models
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Model Predictions")
                predictions = {}
                
                for model_name, model in trained_models.items():
                    if model_name in ['Logistic Regression', 'Support Vector Machine']:
                        pred = model.predict(scaler.transform(input_df))[0]
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(scaler.transform(input_df))[0]
                            confidence = np.max(proba) * 100
                        else:
                            confidence = "N/A"
                    else:
                        pred = model.predict(input_df)[0]
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(input_df)[0]
                            confidence = np.max(proba) * 100
                        else:
                            confidence = "N/A"
                    
                    predictions[model_name] = {'prediction': pred, 'confidence': confidence}
                    
                    if confidence != "N/A":
                        st.write(f"{model_name}: {pred} ({confidence:.1f}% confidence)")
                    else:
                        st.write(f"{model_name}: {pred}")
            
            with col2:
                # Consensus analysis
                pred_counts = {}
                for model_results in predictions.values():
                    pred = model_results['prediction']
                    pred_counts[pred] = pred_counts.get(pred, 0) + 1
                
                consensus_pred = max(pred_counts, key=pred_counts.get)
                consensus_pct = (pred_counts[consensus_pred] / len(predictions)) * 100
                
                st.write("Consensus Analysis")
                st.success(f"Predicted Engagement Level: {consensus_pred}")
                st.write(f"Model Agreement: {consensus_pct:.0f}%")
                
                fig_pred = px.bar(x=list(pred_counts.keys()), y=list(pred_counts.values()),
                                title="Model Consensus",
                                color=list(pred_counts.keys()))
                st.plotly_chart(fig_pred, use_container_width=True)
            
            # Risk assessment
            st.subheader("Risk Assessment")
            
            risk_factors = []
            if weather_condition in ['Heavy Rain', 'Snow', 'Fog']:
                risk_factors.append(f"Adverse weather conditions: {weather_condition}")
            if driver_attention_score < 70:
                risk_factors.append(f"Below optimal driver attention: {driver_attention_score}")
            if visibility_miles < 5:
                risk_factors.append(f"Limited visibility: {visibility_miles} miles")
            if construction_zone:
                risk_factors.append("Construction zone active")
            if emergency_vehicle_present:
                risk_factors.append("Emergency vehicle in vicinity")
            if traffic_density == 'Stop-and-Go':
                risk_factors.append("Stop-and-go traffic conditions")
            if previous_disengagements > 5:
                risk_factors.append(f"Elevated disengagement history: {previous_disengagements}")
            
            if risk_factors:
                st.warning("Identified Risk Factors:")
                for factor in risk_factors:
                    st.write(f"- {factor}")
            else:
                st.success("No significant risk factors identified")
            
            # Recommendations
            st.subheader("System Recommendations")
            
            recommendations = []
            if consensus_pred == 'Low':
                recommendations.append("Manual driving recommended for these conditions")
                recommendations.append("Monitor road conditions continuously")
                recommendations.append("Maintain heightened attention levels")
            elif consensus_pred == 'Medium':
                recommendations.append("Autopilot available with increased monitoring")
                recommendations.append("Stay prepared for immediate manual intervention")
                recommendations.append("Monitor system performance indicators")
            else:
                recommendations.append("Favorable conditions for autopilot operation")
                recommendations.append("System expected to perform reliably")
                recommendations.append("Continue monitoring as conditions evolve")
            
            for rec in recommendations:
                st.write(f"- {rec}")
        
        # Historical comparison
        st.subheader("Historical Performance Comparison")
        
        if st.button("Analyze Similar Historical Cases"):
            sample_conditions = df[
                (df['weather_condition'] == weather_condition) &
                (df['road_type'] == road_type) &
                (df['traffic_density'] == traffic_density)
            ]
            
            if len(sample_conditions) > 0:
                st.write(f"Analysis based on {len(sample_conditions)} similar historical cases:")
                
                engagement_dist = sample_conditions['autopilot_engagement'].value_counts(normalize=True) * 100
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_hist = px.pie(values=engagement_dist.values, 
                                    names=engagement_dist.index,
                                    title="Historical Engagement Distribution")
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    st.write("Historical Performance Summary:")
                    for engagement, pct in engagement_dist.items():
                        st.write(f"{engagement}: {pct:.1f}%")
                    
                    avg_incidents = sample_conditions['safety_incident'].mean() * 100
                    st.write(f"Historical Safety Incident Rate: {avg_incidents:.2f}%")
            else:
                st.write("No historical cases found matching these exact conditions.")

if __name__ == "__main__":
    main()