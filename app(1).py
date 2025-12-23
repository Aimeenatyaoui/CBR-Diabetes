import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- SETTING HALAMAN ---
st.set_page_config(page_title="Diagnosa Diabetes CBR-RF", layout="wide")

# --- LOAD DATA & PARAMETER ---
@st.cache_resource
def load_data():
    # Pastikan file-file ini ada di repository GitHub Anda
    df = pd.read_csv('data_cbr.csv') 
    with open('bobot_rf.pkl', 'rb') as f:
        weights = pickle.load(f)
    with open('norm_params.pkl', 'rb') as f:
        norm_params = pickle.load(f)
    return df, weights, norm_params
    
try:
    df, feature_weights, normalization_params = load_data()
    feature_columns = [col for col in df.columns if col != 'Outcome']
except Exception as e:
    st.error(f"Gagal memuat data: {e}")
    st.stop()

# --- HEADER ---
st.title("🏥 Sistem Deteksi Dini Diabetes Mellitus Tipe 2")
st.info("""
    Sistem ini menggunakan metode **Case-Based Reasoning (CBR)** untuk membandingkan parameter medis Anda 
    dengan basis data kasus terdahulu. Silakan masukkan hasil laboratorium Anda pada panel sebelah kiri.
""")

# --- INPUT USER (SIDEBAR) DENGAN SATUAN ---
st.sidebar.header("📝 Data Laboratorium Pasien")
user_inputs = {}

# Dictionary untuk pemetaan satuan agar lebih profesional
units = {
    "Pregnancies": "Kali",
    "Glucose": "mg/dL",
    "BloodPressure": "mm Hg",
    "SkinThickness": "mm",
    "Insulin": "μU/mL",
    "BMI": "kg/m²",
    "DiabetesPedigreeFunction": "Score",
    "Age": "Tahun"
}

for col in feature_columns:
    p = normalization_params[col]
    unit = units.get(col, "")
    user_inputs[col] = st.sidebar.number_input(
        f"{col} ({unit})", 
        min_value=0.0, 
        value=float(p['min']),
        help=f"Data asli di dataset: {p['min']} - {p['max']}"
    )

# --- PROSES DIAGNOSA ---
if st.sidebar.button("🚀 Mulai Analisis Sistem", use_container_width=True):
    
    # 1. NORMALISASI OTOMATIS
    norm_user_list = []
    for col in feature_columns:
        p = normalization_params[col]
        val_norm = (user_inputs[col] - p['min']) / (p['max'] - p['min']) if (p['max'] - p['min']) != 0 else 0
        norm_user_list.append(val_norm)
    
    user_arr = np.array(norm_user_list)
    weight_arr = np.array([feature_weights[col] for col in feature_columns])
    
    # 2. HITUNG SIMILARITY (Weighted Cosine)
    similarities = []
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

    # 3. TAMPILKAN TABEL PERBANDINGAN
    st.markdown("---")
    st.subheader("🔍 Analisis Perbandingan Kasus Terdahulu")
    st.write("Sistem menemukan 5 kasus dengan pola parameter paling serupa:")
    
    display_df = top_k.copy()
    display_df['Diagnosis'] = display_df['Outcome'].apply(lambda x: "Diabetes" if x == 1 else "Non-Diabetes")
    
    cols_order = ['similarity', 'Diagnosis'] + feature_columns
    st.dataframe(
        display_df[cols_order].style.format({'similarity': "{:.4f}"})
        .background_gradient(subset=['similarity'], cmap='BuGn'),
        use_container_width=True
    )

    # 4. KESIMPULAN NARASI (VERSI AKADEMIS & AMAN)
    st.markdown("---")
    outcomes = top_k['Outcome'].values
    sims = top_k['similarity'].values
    vote_1 = np.sum(sims[outcomes == 1])
    vote_0 = np.sum(sims[outcomes == 0])
    
    final_pred = 1 if vote_1 > vote_0 else 0
    confidence = max(vote_1, vote_0) / np.sum(sims)
    jml_diabetes = np.sum(outcomes == 1)

    st.subheader("📋 Hasil Analisis Sistem")
    
    if final_pred == 1:
        st.error(f"### KATEGORI: TERDETEKSI DIABETES")
        st.write(f"""
            Berdasarkan algoritma *Weighted Cosine Similarity*, data Anda memiliki tingkat kemiripan dominan 
            sebesar **{confidence:.2%}** dengan kelompok pasien pada kategori **Diabetes Mellitus Tipe 2**. 
            Hasil ini diperoleh dari konsistensi pola terhadap basis data kasus yang tersedia.
        """)
    else:
        if jml_diabetes > 0:
            st.warning(f"### KATEGORI: NON-DIABETES (KEMIRIPAN PARSIAL)")
            st.write(f"""
                Sistem mengidentifikasi bahwa kondisi data Anda secara dominan (**{confidence:.2%}**) menyerupai 
                kelompok pasien **Non-Diabetes**. Namun, secara statistik ditemukan pula kemiripan pola dengan 
                {jml_diabetes} kasus pada kategori **Diabetes**. Adanya kemiripan pada kedua kategori ini 
                menunjukkan bahwa parameter medis Anda berada pada ambang batas (*borderline*) yang memerlukan 
                perhatian secara profesional.
            """)
        else:
            st.success(f"### KATEGORI: NON-DIABETES")
            st.write(f"""
                Seluruh kasus terdekat (**{confidence:.2%}**) menunjukkan kesesuaian pola dengan kelompok data 
                pasien pada kategori **Non-Diabetes**. Parameter yang Anda masukkan memiliki konsistensi tinggi 
                dengan basis data kasus sehat dalam sistem.
            """)

    # 5. VISUALISASI BOBOT
    with st.expander("📊 Lihat Bobot Pengaruh Fitur (Random Forest)"):
        st.info("Grafik ini menunjukkan fitur mana yang paling berpengaruh dalam menentukan kemiripan kasus.")
        st.bar_chart(pd.Series(feature_weights).sort_values(), horizontal=True)

else:
    st.write("⬅️ Silakan masukkan angka hasil laboratorium Anda pada sidebar untuk memulai.")

# --- FOOTER / DISCLAIMER ---
st.markdown("---")
st.caption("""
    **Disclaimer:** Sistem ini merupakan alat bantu skrining dini berbasis data statistik (CBR) dan 
    bukan merupakan diagnosa medis final. Seluruh hasil yang ditampilkan harus dikonsultasikan kembali 
    dengan tenaga kesehatan profesional untuk validasi lebih lanjut.
""")
