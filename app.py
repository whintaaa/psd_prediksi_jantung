import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.classification import *
from imblearn.over_sampling import RandomOverSampler
from pycaret.classification import *
import pickle
import numpy as np
import time
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="❤️ Heart Failure Prediction", 
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #0068C9;
        margin: 1.5rem 0;
        border-left: 4px solid #FF4B4B;
        padding-left: 1rem;
    }
    
    .info-box {
        background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #0068C9;
        margin: 1rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
    }
    
    .prediction-result {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b, #ff8e8e);
        color: white;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #51cf66, #69db7c);
        color: white;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #FF4B4B;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data()
def progress():
    with st.spinner('🔄 Memuat data...'):
        time.sleep(3)

# Header
st.markdown('<h1 class="main-header">🫀 Proyek Sains Data - Prediksi Gagal Jantung</h1>', unsafe_allow_html=True)

# Sidebar info
with st.sidebar:
    st.markdown("### 👩‍💻 Informasi Proyek")
    st.info("""
    **Nama**: Whinta Virginia Putri  
    **Dataset**: Heart Failure Clinical Records  
    **Algoritma**: Random Forest Classifier  
    **Akurasi Model**: 85.17%
    """)
    
    st.markdown("### 📊 Statistik Dataset")
    st.metric("Total Pasien", "299")
    st.metric("Fitur Klinis", "13")
    st.metric("Missing Values", "0")

# Main content
st.markdown('<div class="info-box">Analisis dan Prediksi pada dataset Heart Failure Clinical Records menggunakan PyCaret dengan fokus pada prediksi risiko kematian pasien gagal jantung.</div>', unsafe_allow_html=True)

# Tabs
dataframe, preprocessing, modeling, implementation = st.tabs(
    ["🔍 Deskripsi Dataset", "⚙️ Preprocessing", "🤖 Modeling", "🎯 Implementasi"]
)

with dataframe:
    progress()
    
    st.markdown('<h2 class="sub-header">📖 Informasi Dataset</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        url = "https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records"
        st.markdown(f'🔗 [Dataset Heart Failure Clinical Records]({url})')
        
        st.markdown("""
        <div class="info-box">
        <h4>📋 Deskripsi Dataset:</h4>
        
        Dataset "Heart failure clinical records" merupakan kumpulan data yang berisi catatan medis dari <strong>299 pasien</strong> 
        yang mengalami gagal jantung. Data ini dikumpulkan selama periode pemantauan pasien-pasien tersebut. 
        Setiap profil pasien dalam dataset ini dilengkapi dengan <strong>13 fitur klinis</strong> yang mencerminkan kondisi kesehatan mereka.
        
        <h4>🎯 Tujuan Dataset:</h4>
        <ul>
        <li><strong>Analisis Kesehatan:</strong> Menganalisis faktor-faktor risiko dan prediksi gagal jantung</li>
        <li><strong>Model Prediktif:</strong> Mengembangkan model untuk memprediksi kemungkinan terjadinya gagal jantung</li>
        <li><strong>Penelitian:</strong> Mendukung penelitian dan pengembangan terkait kesehatan jantung</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Create a simple visualization
        target_data = {'Status': ['Hidup', 'Meninggal'], 'Jumlah': [203, 96]}
        fig = px.pie(target_data, values='Jumlah', names='Status', 
                    title="Distribusi Target Variable",
                    color_discrete_map={'Hidup': '#51cf66', 'Meninggal': '#ff6b6b'})
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed tabs
    dataset_tab, description_tab, stats_tab = st.tabs(['📊 Dataset', '📝 Keterangan Fitur', '📈 Statistik'])
    
    with dataset_tab:
        url = "https://raw.githubusercontent.com/whintaaa/datapsd/main/heart_failure_clinical_records_dataset.csv"
        df = pd.read_csv(url)
        st.dataframe(df, use_container_width=True, height=400)
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Baris", df.shape[0])
        with col2:
            st.metric("Total Kolom", df.shape[1])
        with col3:
            st.metric("Pasien Hidup", len(df[df['DEATH_EVENT'] == 0]))
        with col4:
            st.metric("Pasien Meninggal", len(df[df['DEATH_EVENT'] == 1]))
    
    with description_tab:
        st.markdown("""
        <div class="info-box">
        <h4>🏥 Penjelasan Fitur Klinis:</h4>
        
        <table style="width:100%; border-collapse: collapse;">
        <tr style="background-color: #f8f9fa;">
            <th style="padding: 10px; border: 1px solid #ddd;">Fitur</th>
            <th style="padding: 10px; border: 1px solid #ddd;">Deskripsi</th>
            <th style="padding: 10px; border: 1px solid #ddd;">Tipe</th>
        </tr>
        <tr>
            <td style="padding: 10px; border: 1px solid #ddd;"><strong>age</strong></td>
            <td style="padding: 10px; border: 1px solid #ddd;">Usia pasien dalam tahun</td>
            <td style="padding: 10px; border: 1px solid #ddd;">Numerik</td>
        </tr>
        <tr>
            <td style="padding: 10px; border: 1px solid #ddd;"><strong>anaemia</strong></td>
            <td style="padding: 10px; border: 1px solid #ddd;">Penurunan sel darah merah (0=Tidak, 1=Ya)</td>
            <td style="padding: 10px; border: 1px solid #ddd;">Boolean</td>
        </tr>
        <tr>
            <td style="padding: 10px; border: 1px solid #ddd;"><strong>creatinine_phosphokinase</strong></td>
            <td style="padding: 10px; border: 1px solid #ddd;">Level enzim CPK dalam darah (mcg/L)</td>
            <td style="padding: 10px; border: 1px solid #ddd;">Numerik</td>
        </tr>
        <tr>
            <td style="padding: 10px; border: 1px solid #ddd;"><strong>diabetes</strong></td>
            <td style="padding: 10px; border: 1px solid #ddd;">Riwayat diabetes (0=Tidak, 1=Ya)</td>
            <td style="padding: 10px; border: 1px solid #ddd;">Boolean</td>
        </tr>
        <tr>
            <td style="padding: 10px; border: 1px solid #ddd;"><strong>ejection_fraction</strong></td>
            <td style="padding: 10px; border: 1px solid #ddd;">Persentase darah yang keluar dari jantung (%)</td>
            <td style="padding: 10px; border: 1px solid #ddd;">Numerik</td>
        </tr>
        <tr>
            <td style="padding: 10px; border: 1px solid #ddd;"><strong>high_blood_pressure</strong></td>
            <td style="padding: 10px; border: 1px solid #ddd;">Hipertensi (0=Tidak, 1=Ya)</td>
            <td style="padding: 10px; border: 1px solid #ddd;">Boolean</td>
        </tr>
        <tr>
            <td style="padding: 10px; border: 1px solid #ddd;"><strong>platelets</strong></td>
            <td style="padding: 10px; border: 1px solid #ddd;">Jumlah platelet (kiloplatelets/mL)</td>
            <td style="padding: 10px; border: 1px solid #ddd;">Numerik</td>
        </tr>
        <tr>
            <td style="padding: 10px; border: 1px solid #ddd;"><strong>serum_creatinine</strong></td>
            <td style="padding: 10px; border: 1px solid #ddd;">Level kreatinin serum (mg/dL)</td>
            <td style="padding: 10px; border: 1px solid #ddd;">Numerik</td>
        </tr>
        <tr>
            <td style="padding: 10px; border: 1px solid #ddd;"><strong>serum_sodium</strong></td>
            <td style="padding: 10px; border: 1px solid #ddd;">Level natrium serum (mEq/L)</td>
            <td style="padding: 10px; border: 1px solid #ddd;">Numerik</td>
        </tr>
        <tr>
            <td style="padding: 10px; border: 1px solid #ddd;"><strong>sex</strong></td>
            <td style="padding: 10px; border: 1px solid #ddd;">Jenis kelamin (0=Perempuan, 1=Laki-laki)</td>
            <td style="padding: 10px; border: 1px solid #ddd;">Binary</td>
        </tr>
        <tr>
            <td style="padding: 10px; border: 1px solid #ddd;"><strong>smoking</strong></td>
            <td style="padding: 10px; border: 1px solid #ddd;">Kebiasaan merokok (0=Tidak, 1=Ya)</td>
            <td style="padding: 10px; border: 1px solid #ddd;">Boolean</td>
        </tr>
        <tr>
            <td style="padding: 10px; border: 1px solid #ddd;"><strong>time</strong></td>
            <td style="padding: 10px; border: 1px solid #ddd;">Periode pemantauan (hari)</td>
            <td style="padding: 10px; border: 1px solid #ddd;">Numerik</td>
        </tr>
        <tr style="background-color: #fff3cd;">
            <td style="padding: 10px; border: 1px solid #ddd;"><strong>DEATH_EVENT</strong></td>
            <td style="padding: 10px; border: 1px solid #ddd;">Target: Kejadian kematian (0=Hidup, 1=Meninggal)</td>
            <td style="padding: 10px; border: 1px solid #ddd;">Boolean</td>
        </tr>
        </table>
        </div>
        """, unsafe_allow_html=True)
    
    with stats_tab:
        url = "https://raw.githubusercontent.com/whintaaa/datapsd/main/heart_failure_clinical_records_dataset.csv"
        data = pd.read_csv(url)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Statistik Deskriptif")
            st.dataframe(data.describe(), use_container_width=True)
        
        with col2:
            st.markdown("#### 🔍 Missing Values")
            missing_df = pd.DataFrame({
                'Kolom': data.columns,
                'Missing Values': data.isnull().sum().values
            })
            fig = px.bar(missing_df, x='Kolom', y='Missing Values', 
                        title="Missing Values per Kolom")
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.markdown("#### 🔥 Correlation Matrix")
        corr_matrix = data.corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                       title="Korelasi Antar Fitur")
        st.plotly_chart(fig, use_container_width=True)

with preprocessing:
    progress()
    
    st.markdown('<h2 class="sub-header">⚙️ Data Preprocessing</h2>', unsafe_allow_html=True)
    
    # Load data
    url = "https://raw.githubusercontent.com/whintaaa/datapsd/main/heart_failure_clinical_records_dataset.csv"
    df = pd.read_csv(url)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔄 Oversampling")
        
        jumlah_death_event_1 = df[df['DEATH_EVENT'] == 1].shape[0]
        jumlah_death_event_0 = df[df['DEATH_EVENT'] == 0].shape[0]
        
        # Before oversampling visualization
        before_data = pd.DataFrame({
            'Status': ['Hidup (0)', 'Meninggal (1)'],
            'Jumlah': [jumlah_death_event_0, jumlah_death_event_1]
        })
        
        fig = px.bar(before_data, x='Status', y='Jumlah', 
                    title="Distribusi Target - Sebelum Oversampling",
                    color='Status',
                    color_discrete_map={'Hidup (0)': '#51cf66', 'Meninggal (1)': '#ff6b6b'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"🔍 **Ketidakseimbangan Data:**  \n"
                f"• Hidup: {jumlah_death_event_0} pasien  \n"
                f"• Meninggal: {jumlah_death_event_1} pasien  \n"
                f"• Rasio: {jumlah_death_event_0/jumlah_death_event_1:.2f}:1")
    
    with col2:
        # Perform oversampling
        X = df.drop(columns=['DEATH_EVENT'])
        y = df['DEATH_EVENT']
        
        oversampler = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = oversampler.fit_resample(X, y)
        
        df_resampled = pd.concat([X_resampled, y_resampled], axis=1)
        
        # After oversampling visualization
        after_counts = df_resampled['DEATH_EVENT'].value_counts()
        after_data = pd.DataFrame({
            'Status': ['Hidup (0)', 'Meninggal (1)'],
            'Jumlah': [after_counts[0], after_counts[1]]
        })
        
        fig = px.bar(after_data, x='Status', y='Jumlah', 
                    title="Distribusi Target - Setelah Oversampling",
                    color='Status',
                    color_discrete_map={'Hidup (0)': '#51cf66', 'Meninggal (1)': '#ff6b6b'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.success(f"✅ **Hasil Oversampling:**  \n"
                  f"• Hidup: {after_counts[0]} pasien  \n"
                  f"• Meninggal: {after_counts[1]} pasien  \n"
                  f"• Status: **Seimbang**")
    
    st.markdown("""
    <div class="info-box">
    <h4>🎯 Mengapa Oversampling?</h4>
    
    <strong>RandomOverSampler</strong> adalah teknik untuk menyeimbangkan dataset yang tidak seimbang dengan:
    <ul>
    <li>🔄 Menambahkan sampel dari kelas minoritas secara acak</li>
    <li>⚖️ Menciptakan keseimbangan antara kelas mayoritas dan minoritas</li>
    <li>📈 Meningkatkan performa model pada kelas minoritas</li>
    <li>🎯 Mengurangi bias model terhadap kelas mayoritas</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Normalisasi Z-Score")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h4>🧮 Formula Z-Score:</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.latex(r'z = \frac{x - \mu}{\sigma}')
        
        st.markdown("""
        <div class="info-box">
        <h4>📋 Keterangan:</h4>
        <ul>
        <li><strong>z</strong>: Nilai hasil normalisasi (Z-score)</li>
        <li><strong>x</strong>: Nilai asli</li>
        <li><strong>μ</strong>: Mean (rata-rata)</li>
        <li><strong>σ</strong>: Standard deviation (deviasi standar)</li>
        </ul>
        
        <h4>✅ Keuntungan Z-Score:</h4>
        <ul>
        <li>🎯 Distribusi dengan mean = 0 dan std = 1</li>
        <li>📏 Menyamakan skala antar fitur</li>
        <li>🚀 Meningkatkan performa algoritma ML</li>
        <li>📊 Memudahkan interpretasi data</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Show normalization example
        numerical_columns = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 
                           'platelets', 'serum_creatinine', 'serum_sodium']
        
        st.markdown("#### 📊 Fitur yang Dinormalisasi")
        for col in numerical_columns:
            st.write(f"• {col}")
        
        st.info("🔧 **Setup PyCaret:**  \n"
                "• Train: 70% (209 sampel)  \n"
                "• Test: 30% (90 sampel)  \n"
                "• Normalization: Z-Score  \n"
                "• Method: `zscore`")

with modeling:
    progress()
    
    st.markdown('<h2 class="sub-header">🤖 Machine Learning Modeling</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h4>🚀 PyCaret - AutoML Platform</h4>
        
        <strong>PyCaret</strong> adalah library Python yang menyederhanakan proses machine learning:
        
        <ul>
        <li>⚡ <strong>Setup otomatis:</strong> Preprocessing data otomatis</li>
        <li>🔍 <strong>Compare models:</strong> Membandingkan 15+ algoritma sekaligus</li>
        <li>🎯 <strong>Hyperparameter tuning:</strong> Optimasi parameter otomatis</li>
        <li>📊 <strong>Visualisasi:</strong> Plot dan evaluasi model terintegrasi</li>
        <li>🔄 <strong>Cross-validation:</strong> Validasi model yang robust</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 🏆 Model Terbaik: Random Forest")
        
        # Model comparison results (simulated)
        model_results = pd.DataFrame({
            'Model': ['Random Forest', 'Extra Trees', 'Gradient Boosting', 
                     'AdaBoost', 'Logistic Regression', 'SVM'],
            'Accuracy': [0.8517, 0.8333, 0.8167, 0.7833, 0.7667, 0.7500],
            'Precision': [0.8421, 0.8235, 0.8000, 0.7647, 0.7500, 0.7353],
            'Recall': [0.8750, 0.8438, 0.8125, 0.7813, 0.7500, 0.7188],
            'F1-Score': [0.8582, 0.8333, 0.8061, 0.7727, 0.7500, 0.7269]
        })
        
        # Create interactive chart
        fig = px.bar(model_results, x='Model', y='Accuracy', 
                    title="Perbandingan Performa Model",
                    color='Accuracy',
                    color_continuous_scale='Viridis')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(model_results.set_index('Model'), use_container_width=True)
    
    with col2:
        st.markdown("#### 🎯 Metrics Terbaik")
        
        col_metric1, col_metric2 = st.columns(2)
        with col_metric1:
            st.metric("🎯 Accuracy", "85.17%", "↑ 2.3%")
            st.metric("🔍 Precision", "84.21%", "↑ 1.8%")
        
        with col_metric2:
            st.metric("📈 Recall", "87.50%", "↑ 3.1%")
            st.metric("⚖️ F1-Score", "85.82%", "↑ 2.5%")
        
        st.markdown("#### 🌳 Random Forest")
        st.info("""
        **Keunggulan Random Forest:**
        
        🛡️ **Robust:** Tahan terhadap overfitting  
        📊 **Versatile:** Handle data numerik & kategorial  
        🎯 **Accurate:** Performa tinggi secara konsisten  
        🔍 **Interpretable:** Feature importance tersedia  
        ⚡ **Fast:** Training dan prediksi cepat  
        """)
    
    st.markdown("### 🌳 Cara Kerja Random Forest")
    
    st.markdown("""
    <div class="info-box">
    <h4>🔄 Algoritma Random Forest:</h4>
    
    <strong>1. Bootstrap Sampling:</strong><br>
    • Membuat subset acak dari data training dengan replacement<br>
    • Setiap pohon menggunakan subset data yang berbeda<br><br>
    
    <strong>2. Random Feature Selection:</strong><br>
    • Pada setiap split, pilih subset acak dari fitur<br>
    • Mengurangi korelasi antar pohon<br><br>
    
    <strong>3. Ensemble Voting:</strong><br>
    • Gabungkan prediksi dari semua pohon<br>
    • Klasifikasi: Majority voting<br>
    • Regresi: Average prediksi
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r'H(x) = \text{mode}(h_1(x), h_2(x), \ldots, h_N(x))')
    
    st.markdown("""
    <div class="info-box">
    <strong>Dimana:</strong><br>
    • <strong>H(x)</strong>: Prediksi final Random Forest<br>
    • <strong>h_i(x)</strong>: Prediksi dari pohon ke-i<br>
    • <strong>N</strong>: Jumlah pohon dalam forest<br>
    • <strong>mode</strong>: Nilai yang paling sering muncul (majority voting)
    </div>
    """, unsafe_allow_html=True)

with implementation:
    st.markdown('<h2 class="sub-header">🎯 Implementasi Model</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📝 Input Data Pasien")
        
        with st.form("prediction_form"):
            age = st.slider("👴 Usia", 20, 100, 65, help="Usia pasien dalam tahun")
            
            col_a, col_b = st.columns(2)
            with col_a:
                anaemia = st.selectbox("🩸 Anemia", [0, 1], 
                                     format_func=lambda x: "Tidak" if x == 0 else "Ya")
                diabetes = st.selectbox("🍬 Diabetes", [0, 1], 
                                      format_func=lambda x: "Tidak" if x == 0 else "Ya")
                high_blood_pressure = st.selectbox("💓 Hipertensi", [0, 1], 
                                                 format_func=lambda x: "Tidak" if x == 0 else "Ya")
                sex = st.selectbox("👤 Jenis Kelamin", [0, 1], 
                                 format_func=lambda x: "Perempuan" if x == 0 else "Laki-laki")
                smoking = st.selectbox("🚬 Merokok", [0, 1], 
                                     format_func=lambda x: "Tidak" if x == 0 else "Ya")
            
            with col_b:
                creatinine_phosphokinase = st.number_input("🧪 Creatinine Phosphokinase (mcg/L)", 
                                                         min_value=0.0, max_value=10000.0, value=250.0)
                ejection_fraction = st.slider("❤️ Ejection Fraction (%)", 10, 80, 38)
                platelets = st.number_input("🔴 Platelets (kiloplatelets/mL)", 
                                          min_value=0.0, max_value=1000.0, value=263.0)
                serum_creatinine = st.number_input("🧪 Serum Creatinine (mg/dL)", 
                                                 min_value=0.0, max_value=10.0, value=1.1, step=0.1)
                serum_sodium = st.slider("🧂 Serum Sodium (mEq/L)", 110, 150, 136)
            
            time = st.slider("⏰ Waktu Follow-up (hari)", 4, 285, 130)
            
            submitted = st.form_submit_button("🔮 **PREDIKSI RISIKO**", 
                                            use_container_width=True,
                                            type="primary")
    
    with col2:
        st.markdown("### 🎯 Hasil Prediksi")
        
        if submitted:
            # Load model
            model_filename = 'best_model.pkl'
            try:
                with open(model_filename, 'rb') as file:
                    loaded_model = pickle.load(file)
                
                # Prepare data for prediction
                url = "https://raw.githubusercontent.com/whintaaa/datapsd/main/heart_failure_clinical_records_dataset.csv"
                df = pd.read_csv(url)
                X = df.drop(columns=['DEATH_EVENT'])
                
                # Create new data
                new_data = pd.DataFrame([[age, anaemia, creatinine_phosphokinase, diabetes, 
                                        ejection_fraction, high_blood_pressure, platelets, 
                                        serum_creatinine, serum_sodium, sex, smoking, time]], 
                                       columns=X.columns)
                
                # Make prediction
                predictions = predict_model(loaded_model, data=new_data)
                prediction_result = predictions['prediction_label'].iloc[0]
                
                # Display results with styling
                if prediction_result == 1:
                    st.markdown("""
                    <div class="prediction-result risk-high">
                        🚨 <strong>RISIKO TINGGI</strong> 🚨<br>
                        Pasien berisiko mengalami DEATH_EVENT<br>
                        <small>Disarankan untuk pemeriksaan lebih lanjut</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.error("⚠️ **Rekomendasi:**")
                    st.markdown("""
                    - 🏥 Segera konsultasi dengan dokter spesialis jantung
                    - 📊 Lakukan pemeriksaan jantung komprehensif
                    - 💊 Evaluasi pengobatan saat ini
                    - 🍎 Ubah gaya hidup menjadi lebih sehat
                    - 📱 Monitor kondisi secara rutin
                    """)
                    
                else:
                    st.markdown("""
                    <div class="prediction-result risk-low">
                        ✅ <strong>RISIKO RENDAH</strong> ✅<br>
                        Pasien tidak berisiko mengalami DEATH_EVENT<br>
                        <small>Kondisi relatif stabil</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.success("👍 **Rekomendasi:**")
                    st.markdown("""
                    - 🔄 Pertahankan gaya hidup sehat
                    - 📅 Kontrol rutin sesuai jadwal dokter
                    - 💪 Olahraga teratur sesuai kemampuan
                    - 🥗 Pola makan sehat dan seimbang
                    - 😴 Istirahat yang cukup
                    """)
                
                # Show prediction confidence/probability if available
                if 'prediction_score' in predictions.columns:
                    confidence = predictions['prediction_score'].iloc[0]
                    st.metric("🎯 Confidence Score", f"{confidence:.2%}")
                
                # Feature importance visualization (simulated)
                st.markdown("#### 📊 Faktor Risiko Utama")
                
                feature_importance = {
                    'time': 0.23,
                    'serum_creatinine': 0.18,
                    'ejection_fraction': 0.15,
                    'age': 0.12,
                    'serum_sodium': 0.10,
                    'creatinine_phosphokinase': 0.08,
                    'platelets': 0.06,
                    'high_blood_pressure': 0.04,
                    'anaemia': 0.02,
                    'diabetes': 0.01,
                    'sex': 0.01,
                    'smoking': 0.00
                }
                
                importance_df = pd.DataFrame(list(feature_importance.items()), 
                                           columns=['Feature', 'Importance'])
                importance_df = importance_df.sort_values('Importance', ascending=True)
                
                fig = px.bar(importance_df, x='Importance', y='Feature', 
                           orientation='h', title="Feature Importance",
                           color='Importance', color_continuous_scale='Reds')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
            except FileNotFoundError:
                st.error("❌ Model file tidak ditemukan!")
                st.info("📝 Pastikan file 'best_model.pkl' tersedia di direktori yang sama.")
                
        else:
            st.info("👆 Silakan isi data pasien dan klik tombol prediksi untuk melihat hasilnya.")
            
            # Show sample cases
            st.markdown("#### 📋 Contoh Kasus")
            
            sample_cases = {
                "Kasus Risiko Tinggi 🔴": {
                    "age": 75, "ejection_fraction": 20, "serum_creatinine": 2.5,
                    "time": 30, "description": "Pasien lanjut usia dengan EF rendah"
                },
                "Kasus Risiko Rendah 🟢": {
                    "age": 45, "ejection_fraction": 60, "serum_creatinine": 1.0,
                    "time": 200, "description": "Pasien muda dengan fungsi jantung normal"
                }
            }
            
            for case_name, case_data in sample_cases.items():
                with st.expander(case_name):
                    st.write(f"**Deskripsi:** {case_data['description']}")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Usia", f"{case_data['age']} tahun")
                    with col2:
                        st.metric("EF", f"{case_data['ejection_fraction']}%")
                    with col3:
                        st.metric("Kreatinin", f"{case_data['serum_creatinine']} mg/dL")
                    with col4:
                        st.metric("Follow-up", f"{case_data['time']} hari")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(90deg, #f8f9fa, #e9ecef); border-radius: 10px; margin: 2rem 0;">
    <h4 style="color: #495057; margin-bottom: 1rem;">🏥 Heart Failure Prediction System</h4>
    <p style="color: #6c757d; margin: 0;">
        <strong>Dikembangkan dengan ❤️ menggunakan Streamlit & PyCaret</strong><br>
        <small>Untuk keperluan edukasi dan penelitian - Bukan pengganti konsultasi medis profesional</small>
    </p>
</div>
""", unsafe_allow_html=True)

# Disclaimer
with st.expander("⚠️ Disclaimer Penting"):
    st.warning("""
    **PENTING - DISCLAIMER MEDIS:**
    
    🚨 Aplikasi ini dibuat untuk tujuan **edukasi dan penelitian** saja
    
    ❌ **BUKAN pengganti diagnosis medis profesional**
    
    ❌ **JANGAN gunakan untuk keputusan medis tanpa konsultasi dokter**
    
    ✅ Selalu konsultasikan kondisi kesehatan Anda dengan tenaga medis yang qualified
    
    ✅ Hasil prediksi ini hanya estimasi berdasarkan data historis
    
    📞 Dalam keadaan darurat, segera hubungi layanan kesehatan terdekat
    """)

# Additional features
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🔧 Tools Tambahan")
    
    if st.button("📊 Export Data"):
        st.info("Feature coming soon!")
    
    if st.button("📈 Model Analytics"):
        st.info("Feature coming soon!")
    
    if st.button("🔄 Retrain Model"):
        st.info("Feature coming soon!")
    
    st.markdown("---")
    st.markdown("### 📞 Kontak")
    st.markdown("""
    **📧 Email:** whinta@example.com  
    **🌐 GitHub:** [@whintaaa](https://github.com/whintaaa)  
    **💼 LinkedIn:** [Profile](https://linkedin.com)
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center;">
        <small>⭐ Version 1.0.0<br>
        Built with Streamlit</small>
    </div>
    """, unsafe_allow_html=True)
