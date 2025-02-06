import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing

# 1. Upload Excel file
st.title("Prediksi Harga Beras Menggunakan Berbagai Metode Moving Average & Eksponential Smoothing ")
uploaded_file = st.file_uploader("Unggah file Excel", type=["xlsx"])

if uploaded_file is not None:
    # 2. Load Excel file
    data = pd.read_excel(uploaded_file)
    
    # 3. Tampilkan daftar jenis beras yang tersedia
    st.write("Kolom harga beras yang tersedia:")
    price_columns = data.columns[2:]  # Kolom harga mulai dari kolom ke-3
    for i, col in enumerate(price_columns, start=1):
        st.write(f"{i}. {col}")
    
    # 4. Pilih jenis beras
    choice = st.selectbox("Pilih nomor kolom beras yang ingin dianalisis:", range(1, len(price_columns) + 1))
    price_column = price_columns[choice - 1]
    
    # 5. Proses data
    data['day_of_year'] = np.arange(1, len(data) + 1)
    data = data[['month', 'Tanggal', 'day_of_year', price_column]].rename(columns={price_column: 'price'})
    
    # 6. Hapus data ekstrem (harga < 12000)
    data = data[data['price'] >= 12000]
    
    # 7. Pastikan ada cukup data setelah filter harga
    if data.empty:
        st.error("Setelah memfilter harga < 12000, tidak ada data yang tersisa. Periksa kembali dataset.")
    else:
        # 8. Tampilkan ringkasan data
        summary_data = {
            'Rata-Rata Harga': [data['price'].mean()],
            'Harga Minimum': [data['price'].min()],
            'Harga Maksimum': [data['price'].max()],
            'Jumlah Data': [len(data)]
        }
        summary_table = pd.DataFrame(summary_data)
        st.subheader("Ringkasan Data")
        st.write(summary_table)

        # 9. Pilih metode prediksi
        prediction_method = st.selectbox(
            "Pilih metode prediksi:",
            ["Single Moving Average (SMA)",
             "Double Moving Average (DMA)",
             "Triple Moving Average (TMA)",
             "Single Exponential Smoothing (SES)",
             "Double Exponential Smoothing (DES)",
             "Triple Exponential Smoothing (TES)"]
        )
        
        # 10. Parameter sesuai metode yang dipilih
        horizon = st.selectbox("Pilih horizon prediksi:", [7, 14, 30])
        
        if "Moving Average" in prediction_method:
            window_size = st.slider("Pilih window size:", min_value=3, max_value=60, value=30)
        
        if "Exponential" in prediction_method:
            if "Single" in prediction_method:
                alpha = st.select_slider("Pilih parameter alpha:", options=[0.2, 0.5, 0.8])
            elif "Double" in prediction_method:
                alpha = st.select_slider("Pilih parameter alpha:", options=[0.2, 0.5, 0.8])
                beta = st.select_slider("Pilih parameter beta:", options=[0.2, 0.5, 0.8])
            else:  # Triple
                alpha = st.select_slider("Pilih parameter alpha:", options=[0.2, 0.5, 0.8])
                beta = st.select_slider("Pilih parameter beta:", options=[0.1, 0.3, 0.5])
                gamma = st.select_slider("Pilih parameter gamma:", options=[0.1, 0.3, 0.5])

        # 11. Fungsi untuk berbagai metode MA
        def calculate_ma(values, window):
            return pd.Series(values).rolling(window=window).mean()

        def double_ma(values, window):
            single_ma = calculate_ma(values, window)
            double_ma = calculate_ma(single_ma, window)
            
            a = 2 * single_ma - double_ma
            b = (2 / (window - 1)) * (single_ma - double_ma)
            
            return single_ma, double_ma, a, b

        def triple_ma(values, window):
            single_ma = calculate_ma(values, window)
            double_ma = calculate_ma(single_ma, window)
            triple_ma = calculate_ma(double_ma, window)
            
            a = 3 * single_ma - 3 * double_ma + triple_ma
            b = (3 / (2 * (window - 1))) * ((6 - 5) * single_ma - 2 * (5 - 4) * double_ma + (4 - 3) * triple_ma)
            c = (2 / ((window - 1) * (window - 2))) * (single_ma - 2 * double_ma + triple_ma)
            
            return single_ma, double_ma, triple_ma, a, b, c

        # 12. Perhitungan prediksi berdasarkan metode yang dipilih
        predictions = []
        fitted_values = None
        
        if prediction_method == "Single Moving Average (SMA)":
            data['MA'] = calculate_ma(data['price'], window_size)
            last_ma = data['MA'].dropna().iloc[-1]
            predictions = [last_ma] * horizon
            fitted_values = data['MA']

        elif prediction_method == "Double Moving Average (DMA)":
            single_ma, double_ma, a, b = double_ma(data['price'], window_size)
            data['DMA'] = double_ma
            last_a = a.iloc[-1]
            last_b = b.iloc[-1]
            predictions = [last_a + last_b * h for h in range(1, horizon + 1)]
            fitted_values = double_ma

        elif prediction_method == "Triple Moving Average (TMA)":
            single_ma, double_ma, triple_ma, a, b, c = triple_ma(data['price'], window_size)
            data['TMA'] = triple_ma
            last_a = a.iloc[-1]
            last_b = b.iloc[-1]
            last_c = c.iloc[-1]
            predictions = [last_a + last_b * h + 0.5 * last_c * h ** 2 for h in range(1, horizon + 1)]
            fitted_values = triple_ma

        elif prediction_method == "Single Exponential Smoothing (SES)":
            model = SimpleExpSmoothing(data['price'])
            fit = model.fit(smoothing_level=alpha, optimized=False)
            predictions = fit.forecast(horizon).tolist()
            fitted_values = fit.fittedvalues

        elif prediction_method == "Double Exponential Smoothing (DES)":
            model = ExponentialSmoothing(data['price'], trend='add')
            fit = model.fit(smoothing_level=alpha, smoothing_trend=beta, optimized=False)
            predictions = fit.forecast(horizon).tolist()
            fitted_values = fit.fittedvalues

        elif prediction_method == "Triple Exponential Smoothing (TES)":
            # Menggunakan seasonal_periods=7 untuk pola mingguan
            model = ExponentialSmoothing(data['price'], trend='add', seasonal='add', seasonal_periods=7)
            fit = model.fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma, optimized=False)
            predictions = fit.forecast(horizon).tolist()
            fitted_values = fit.fittedvalues

        # 13. Tampilkan hasil prediksi
        st.subheader("Hasil Prediksi")
        
        # Get the actual prices for comparison
        actual_prices = data['price'].iloc[-horizon:].tolist()
        
        # Create DataFrame for results
        results = pd.DataFrame({
            'Hari ke-': list(range(len(data) - horizon + 1, len(data) + 1)),  # Historical days
            'Harga Asli': actual_prices,
            'Fitted Values': fitted_values.iloc[-horizon:] if fitted_values is not None else [None] * horizon
        })
        
        # Add prediction results with adjusted day numbers
        prediction_results = pd.DataFrame({
            'Hari ke-': list(range(len(data) + 1, len(data) + horizon + 1)),  # Future days
            'Harga Asli': [None] * horizon,  # No actual prices for future dates
            'Prediksi': predictions
        })
        
        # Combine historical and prediction results
        results = pd.concat([results, prediction_results], ignore_index=True)
        
        # Format the numeric columns
        numeric_columns = ['Harga Asli', 'Fitted Values', 'Prediksi']
        for col in numeric_columns:
            if col in results.columns:
                results[col] = results[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "-")
        
        st.write(results)

        # 14. Plot hasil
        plt.clf()
        plt.figure(figsize=(14, 6))
        plt.plot(data['day_of_year'], data['price'], label='Harga Asli', color='black')
        
        if fitted_values is not None:
            plt.plot(data['day_of_year'], fitted_values, label=f'Fitted Values', color='blue')
        
        plt.plot(
            range(len(data) + 1, len(data) + 1 + horizon),
            predictions,
            color='red',
            linestyle='--',
            label=f'Prediksi {prediction_method}'
        )
        
        plt.title(f'Hasil Prediksi Harga Beras - {price_column}')
        plt.xlabel('Hari ke-')
        plt.ylabel('Harga')
        plt.legend()
        st.pyplot(plt)