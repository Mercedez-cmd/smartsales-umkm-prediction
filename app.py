import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

st.set_page_config(
    page_title="SmartSales UMKM - Prediksi Penjualan",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_model():
    bundle = joblib.load("smartsales_model.pth")
    return bundle["model"], bundle["features"]

model, feature_columns = load_model()

st.title("📊 SmartSales: Sistem Prediksi Penjualan Harian UMKM")
st.markdown(
    "Aplikasi berbasis **Machine Learning (Random Forest Regressor)** untuk membantu "
    "UMKM kuliner merencanakan stok bahan baku harian dan menekan risiko *food waste*."
)
st.divider()

col_input, col_result = st.columns([1.1, 1], gap="large")

with col_input:
    st.subheader("📝 Input Data Operasional & Histori")

    tanggal = st.date_input("Tanggal Prediksi Penjualan", datetime.date.today())
    cabang = st.selectbox("Cabang / Gerai UMKM", options=[1, 2, 3], format_func=lambda x: f"Gerai #{x}")

    menu_map = {
        "Tahu Crispy (Prepared Foods)": "PREPARED FOODS",
        "Sosis Crispy (Meats)": "MEATS",
        "Otak-Otak & Ayam (Poultry)": "POULTRY",
        "Paket Mix / Camilan (Deli)": "DELI"
    }
    menu_label = st.selectbox("Kategori Menu UMKM", list(menu_map.keys()))
    family_selected = menu_map[menu_label]

    promo = st.radio(
        "Status Promo / Diskon Hari Ini",
        options=[0, 1],
        format_func=lambda x: "Ada Promo (Bundling/Diskon)" if x == 1 else "Harga Normal (Tanpa Promo)",
        horizontal=True
    )

    st.markdown("**Data Histori Penjualan (Porsi/Unit):**")
    lag_1 = st.number_input("Penjualan Kemarin (H-1)", min_value=0, max_value=5000, value=145)
    lag_7 = st.number_input("Penjualan 7 Hari Lalu (H-7)", min_value=0, max_value=5000, value=160)
    rolling_mean_7 = st.number_input("Rata-rata Penjualan 7 Hari Terakhir", min_value=0.0, max_value=5000.0, value=150.0)

    prediksi_btn = st.button("🚀 Hitung Prediksi Penjualan", type="primary", use_container_width=True)

with col_result:
    st.subheader("📈 Hasil Prediksi & Rekomendasi Stok")

    if prediksi_btn:
        day_of_week = tanggal.weekday()
        month = tanggal.month
        is_weekend = 1 if day_of_week in [5, 6] else 0

        input_dict = {
            "store_nbr": cabang,
            "onpromotion": promo,
            "day_of_week": day_of_week,
            "month": month,
            "is_weekend": is_weekend,
            "lag_1": lag_1,
            "lag_7": lag_7,
            "rolling_mean_7": rolling_mean_7,
        }

        for col in feature_columns:
            if col.startswith("family_"):
                cat_name = col.replace("family_", "")
                input_dict[col] = 1 if cat_name == family_selected else 0

        input_df = pd.DataFrame([input_dict]).reindex(columns=feature_columns, fill_value=0)

        hasil_prediksi = model.predict(input_df)[0]
        prediksi_bulat = max(0, int(round(hasil_prediksi)))
        stok_aman = int(np.ceil(prediksi_bulat * 1.05))

        m1, m2 = st.columns(2)
        m1.metric("Estimasi Terjual", f"{prediksi_bulat} Porsi", delta=f"{prediksi_bulat - lag_1:+d} vs Kemarin")
        m2.metric("Rekomendasi Siapkan Stok", f"{stok_aman} Porsi", delta="Buffer +5% Aman")

        st.success(
            f"**Analisis Operasional:** Pada tanggal **{tanggal.strftime('%d %B %Y')}**, "
            f"permintaan menu **{menu_label}** diprediksi mencapai **{prediksi_bulat} porsi**. "
            f"Disarankan menyiapkan bahan baku untuk **{stok_aman} porsi** guna menghindari kehabisan stok maupun sisa berlebih."
        )

        chart_df = pd.DataFrame({
            "Keterangan": ["Penjualan H-7", "Penjualan Kemarin (H-1)", "Rata-rata 7 Hari", "Prediksi Hari Ini"],
            "Jumlah Porsi": [lag_7, lag_1, rolling_mean_7, prediksi_bulat]
        }).set_index("Keterangan")
        st.bar_chart(chart_df)
    else:
        st.info("Silakan lengkapi parameter di sebelah kiri, lalu klik **Hitung Prediksi Penjualan**.")

st.divider()
st.caption("SmartSales UMKM • Dikembangkan oleh Abimanyu Poendiarta Ramadhan (202343501738) • CRISP-DM Machine Learning")
