import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- SETTING HALAMAN ---
st.set_page_config(page_title="Diagnosa Diabetes CBR-RF", layout="wide")

# --- LOAD DATA & PARAMETER (Sesuaikan path file-mu) ---
@st.cache_resource
def load_data():
    # Ganti path ini sesuai lokasi file kamu di Drive
    df = pd.read_csv('/content/drive/MyDrive/skripsi/preprocessed_data.csv')
    with open('/content/drive/MyDrive/skripsi/feature_weights.pkl', 'rb') as f:
        weights = pickle.load(f)
    with open('/content/drive/MyDrive/skripsi/normalization_params.pkl', 'rb') as f:
        norm_params = pickle.load(f)
    return df, weights, norm_params

try:
    df, feature_weights, normalization_params = load_data()
    feature_columns = [col for col in df.columns if col != 'Outcome']
except:
    st.error("File data tidak ditemukan! Pastikan Drive sudah di-mount dan path filenya benar.")
    st.stop()

# --- JUDUL ---
st.title("🏥 Sistem Pakar Deteksi Dini Diabetes")
st.markdown("---")

# --- INPUT USER (SIDEBAR) ---
st.sidebar.header("📝 Input Data Pasien")
user_inputs = {}

for col in feature_columns:
    # Kita ambil nilai min/max asli agar user tidak bingung inputnya
    # (Input angka normal, nanti sistem yang menormalisasi)
    p = normalization_params[col]
    user_inputs[col] = st.sidebar.number_input(f"{col}", value=float(p['min']))

# --- TOMBOL DIAGNOSA ---
if st.sidebar.button("Mulai Diagnosa"):
    
    # 1. PROSES PREPROCESSING INPUT USER (Sesuai tahap 4 di programmu)
    norm_user_list = []
    for col in feature_columns:
        p = normalization_params[col]
        # Rumus: (x - min) / (max - min)
        val_norm = (user_inputs[col] - p['min']) / (p['max'] - p['min']) if (p['max'] - p['min']) != 0 else 0
        norm_user_list.append(val_norm)
    
    user_arr = np.array(norm_user_list)
    weight_arr = np.array([feature_weights[col] for col in feature_columns])
    
    # 2. PROSES CBR (Weighted Cosine Similarity)
    similarities = []
    for idx, row in df.iterrows():
        case_old_arr = row[feature_columns].values
        
        # Logika: weighted_new vs weighted_old
        weighted_user = user_arr * weight_arr
        weighted_case = case_old_arr * weight_arr
        
        dot_product = np.dot(weighted_user, weighted_case)
        norm_u = np.linalg.norm(weighted_user)
        norm_c = np.linalg.norm(weighted_case)
        
        sim = dot_product / (norm_u * norm_c) if (norm_u * norm_c) != 0 else 0
        similarities.append(sim)
    
    # Masukkan hasil sim ke dataframe
    df_result = df.copy()
    df_result['similarity'] = similarities
    top_k = df_result.sort_values('similarity', ascending=False).head(5)
    
    # 3. TAMPILKAN HASIL
    st.subheader("🔍 Hasil Analisis Kasus Terdekat")
    
    # Tampilkan tabel Top 5 kemiripan
    st.dataframe(top_k[['Outcome', 'similarity']].style.highlight_max(axis=0))

    # 4. PREDIKSI AKHIR (Weighted Voting)
    outcomes = top_k['Outcome'].values
    sims = top_k['similarity'].values
    
    vote_1 = np.sum(sims[outcomes == 1])
    vote_0 = np.sum(sims[outcomes == 0])
    
    final_pred = 1 if vote_1 > vote_0 else 0
    confidence = max(vote_1, vote_0) / np.sum(sims)
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Tingkat Keyakinan (Confidence)", f"{confidence:.2%}")
    
    with col2:
        if final_pred == 1:
            st.error("### HASIL: BERISIKO DIABETES")
        else:
            st.success("### HASIL: TIDAK BERISIKO DIABETES")
            
    # Visualisasi Tambahan (Opsional)
    st.bar_chart(pd.Series(feature_weights, name="Bobot Fitur (RF)"))
