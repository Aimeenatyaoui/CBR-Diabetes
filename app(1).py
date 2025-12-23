import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- SETTING HALAMAN ---
st.set_page_config(page_title="Diagnosa Diabetes CBR-RF", layout="wide")

# --- LOAD DATA & PARAMETER ---
@st.cache_resource
def load_data():
    df = pd.read_csv('data_cbr.csv') 
    with open('bobot_rf.pkl', 'rb') as f:
        weights = pickle.load(f)
    with open('norm_params.pkl', 'rb') as f:
        norm_params = pickle.load(f)
    return df, weights, norm_params
    
try:
    df, feature_weights, normalization_params = load_data()
    feature_columns = [col for col in df.columns if col != 'Outcome']
except:
    st.error("File data tidak ditemukan! Pastikan file sudah ada di GitHub.")
    st.stop()

# --- HEADER ---
st.title("🏥 Sistem Deteksi Dini Diabetes Mellitus Tipe 2")
st.info("Masukkan data klinis Anda pada panel di sebelah kiri. Sistem akan otomatis melakukan normalisasi dan membandingkan kondisi Anda dengan basis kasus.")

# --- INPUT USER (SIDEBAR) ---
st.sidebar.header("📝 Data Pasien Baru")
user_inputs = {}

# MEMPERBAIKI KOLOM INPUT: Sekarang user memasukkan angka asli (bukan angka 0-1)
for col in feature_columns:
    p = normalization_params[col]
    # Kita gunakan nilai asli min dan max dari parameter normalisasi
    user_inputs[col] = st.sidebar.number_input(
        f"{col}", 
        min_value=None, # Kita hilangkan batasan ketat agar user lebih bebas input
        value=float(p['min']), # Nilai awal dimulai dari nilai terkecil asli
        help=f"Nilai minimal di dataset: {p['min']}, maksimal: {p['max']}"
    )

# --- PROSES DIAGNOSA ---
if st.sidebar.button("🚀 Mulai Diagnosa", use_container_width=True):
    
    # 1. PROSES NORMALISASI OTOMATIS (User tidak perlu hitung sendiri)
    norm_user_list = []
    for col in feature_columns:
        p = normalization_params[col]
        # Rumus normalisasi: (Nilai_Input - Min_Asli) / (Max_Asli - Min_Asli)
        val_norm = (user_inputs[col] - p['min']) / (p['max'] - p['min']) if (p['max'] - p['min']) != 0 else 0
        norm_user_list.append(val_norm)
    
    user_arr = np.array(norm_user_list)
    weight_arr = np.array([feature_weights[col] for col in feature_columns])
    
    # 2. HITUNG SIMILARITY (Weighted Cosine)
    similarities = []
    # Bandingkan input yang sudah dinormalisasi tadi dengan dataset (yang juga sudah ternormalisasi)
    for idx, row in df.iterrows():
        case_old_arr = row[feature_columns].values
        weighted_user = user_arr * weight_arr
        weighted_case = case_old_arr * weight_arr
        
        dot_product = np.dot(weighted_user, weighted_case)
        norm_u = np.linalg.norm(weighted_user)
        norm_c = np.linalg.norm(weighted_case)
        
        sim = dot_product / (norm_u * norm_c) if (norm_u * norm_c) != 0 else 0
        similarities.append(sim)
    
    df_result = df.copy()
    df_result['similarity'] = similarities
    top_k = df_result.sort_values('similarity', ascending=False).head(5)

    # 3. TAMPILKAN HASIL
    st.markdown("---")
    st.subheader("🔍 Analisis Perbandingan Kasus")
    
    display_df = top_k.copy()
    display_df['Diagnosis'] = display_df['Outcome'].apply(lambda x: "Diabetes" if x == 1 else "Non-Diabetes")
    
    # Menampilkan tabel (Data di tabel tetap data normalisasi agar sesuai hitungan similarity)
    cols_order = ['similarity', 'Diagnosis'] + feature_columns
    st.dataframe(
        display_df[cols_order].style.format({'similarity': "{:.4f}"})
        .background_gradient(subset=['similarity'], cmap='BuGn'),
        use_container_width=True
    )

    # 4. KESIMPULAN NARASI
    st.markdown("---")
    outcomes = top_k['Outcome'].values
    sims = top_k['similarity'].values
    vote_1 = np.sum(sims[outcomes == 1])
    vote_0 = np.sum(sims[outcomes == 0])
    
    final_pred = 1 if vote_1 > vote_0 else 0
    confidence = max(vote_1, vote_0) / np.sum(sims)

    st.subheader("📋 Kesimpulan Diagnosis")
    status_teks = "terdeteksi Diabetes Mellitus Tipe 2" if final_pred == 1 else "tidak terdeteksi Diabetes Mellitus Tipe 2"
    
    narasi = f"""
    Berdasarkan hasil perhitungan *Weighted Cosine Similarity*, dari ke-5 kasus paling mirip menunjukkan bahwa 
    kasus Anda **{status_teks}** dengan tingkat keyakinan sebesar **{confidence:.2%}**.
    """
    
    if final_pred == 1:
        st.error(narasi)
    else:
        st.success(narasi)

else:
    st.write("Silakan masukkan data klinis Anda di sidebar dan klik tombol **Mulai Diagnosa**.")
