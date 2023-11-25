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


# display
st.set_page_config(page_title="WhintaVP", )

@st.cache_data()
def progress():
    with st.spinner('Wait for it...'):
        time.sleep(5)

st.title("Proyek Sains Data")
st.write("Analisis dan Prediksi pada dataset Heart failure clinical records Menggunakan Pycaret")
st.write('Nama : Whinta Virginia Putri')
st.write('NIM : 210411100047')

dataframe, preporcessing, modeling, implementation = st.tabs(
    ["Deskripsi", "Prepocessing", "Modeling", "Implementation"])

with dataframe:
    progress()
    url = "https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records"
    st.markdown(f'[Dataset Heart failure clinical records]({url})')
    st.write("""
                Dataset "Heart failure clinical records" merupakan kumpulan data yang berisi catatan medis dari 299 pasien yang mengalami gagal jantung. Data ini dikumpulkan selama periode pemantauan pasien-pasien tersebut. Setiap profil pasien dalam dataset ini dilengkapi dengan 13 fitur klinis yang mencerminkan kondisi kesehatan mereka. Di bawah ini, saya akan menjelaskan deskripsi dari dataset ini serta tujuan utamanya:

                #### Deskripsi Dataset:
                1. Jumlah Sampel: Dataset ini berisi informasi dari 299 pasien yang mengalami gagal jantung dan tidak ada missing values.

                2. Fitur Klinis: Setiap pasien dalam dataset ini memiliki 13 fitur klinis yang mencakup berbagai aspek dari kesehatan mereka. Beberapa contoh fitur klinis yang mungkin termasuk dalam dataset ini adalah Usia, tekanan darah, kadar serum kreatinin, kadar serum natrium, kadar serum kalium, ejection fraction (fraksi ejeksi), jenis kelamin pasien, dan lain sebagainya.  dibawah ini adalah fitur pada dataset:
                    'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
                    'ejection_fraction', 'high_blood_pressure', 'platelets',
                    'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time',
                    'DEATH_EVENT'

                3. Dataset ini memiliki target pada kolom Death Event. Fitur target "death event" adalah sebuah fitur yang digunakan untuk menunjukkan apakah pasien mengalami kematian selama periode pemantauan atau tidak. Fitur ini bersifat boolean, yang berarti nilainya hanya dapat berupa dua kemungkinan: 1 / True (benar) atau 0 / False (salah). Jumlah Target dengan nilai 1 Berjumlah 96 dan nilai 0 berjumlah 203.

                4. Tujuan Dataset: Dataset ini memiliki beberapa tujuan utama, antara lain:

                a. Analisis dan Penelitian Kesehatan: Data ini dapat digunakan untuk menganalisis faktor-faktor risiko dan prediksi gagal jantung, serta untuk memahami hubungan antara berbagai fitur klinis dan kondisi pasien.
                
                b. Pengembangan Model Prediktif: Dataset ini dapat digunakan untuk mengembangkan model prediktif yang dapat memprediksi kemungkinan terjadinya gagal jantung pada pasien berdasarkan fitur-fitur klinis mereka. Hal ini dapat membantu tenaga medis dalam melakukan tindakan pencegahan yang lebih tepat waktu.

                Tujuan utama dari dataset ini adalah meningkatkan pemahaman tentang gagal jantung, membantu dalam pengembangan metode prediktif, serta mendukung penelitian dan pengembangan terkait kesehatan jantung. Data ini menjadi dasar penting untuk menjalankan berbagai analisis dan penelitian dalam upaya untuk meningkatkan diagnosis, perawatan, dan pencegahan penyakit gagal jantung.""")
    dataset, ket, eks = st.tabs(['Dataset', 'Keterangan Dataset','Eksplorasi Dataset / Satistik Dataset'])
    with ket:
        url = "https://raw.githubusercontent.com/whintaaa/datapsd/main/heart_failure_clinical_records_dataset.csv"
        df = pd.read_csv(url)
        st.dataframe(df.columns)
        st.write("""
            Fitur-fitur klinis dalam dataset "Heart Failure Clinical Records" adalah sebagai berikut:

            1. **Usia (age)**: Ini adalah usia pasien dalam tahun. Fitur ini memberikan informasi tentang berapa usia pasien yang mengalami gagal jantung. Usia seringkali menjadi faktor penting dalam menilai risiko dan prognosis penyakit jantung. Fitur ini bertipe data numerik.

            2. **Anemia (anaemia)**: Ini adalah fitur boolean yang menunjukkan apakah pasien mengalami penurunan jumlah sel darah merah atau kadar hemoglobin. Nilai 1 / "true" menunjukkan kehadiran anemia, sementara 0 / "false" menunjukkan ketiadaan anemia.

            3. **Kreatinin Fosfokinase (CPK)**: Ini adalah tingkat enzim CPK dalam darah, diukur dalam mikrogram per liter (mcg/L). Tingkat CPK dalam darah dapat memberikan indikasi adanya kerusakan otot atau jaringan jantung. Ini adalah indikator penting dalam penilaian kondisi jantung. Fitur ini bertipe data numerik.

            4. **Diabetes**: Ini adalah fitur boolean yang menunjukkan apakah pasien menderita diabetes atau tidak. Nilai 1 / "true" menunjukkan keberadaan diabetes, sementara 0 / "false" menunjukkan ketiadaan diabetes. Diabetes merupakan faktor risiko yang signifikan dalam perkembangan penyakit jantung.

            5. **Fraksi Ejeksi (Ejection Fraction)**: Ini adalah persentase darah yang meninggalkan jantung pada setiap kontraksi. Fraksi ejeksi ini dinyatakan dalam persentase. Ini adalah ukuran penting dalam menilai kemampuan jantung untuk memompa darah dan dapat memberikan informasi tentang fungsi jantung. Fitur ini bertipe data numerik.

            6. **Tekanan Darah Tinggi (High Blood Pressure)**: Ini adalah fitur boolean yang menunjukkan apakah pasien memiliki hipertensi atau tidak. Nilai 1 / "true" menunjukkan keberadaan tekanan darah tinggi, sementara 2 / "false" menunjukkan ketiadaan tekanan darah tinggi. Tekanan darah tinggi adalah faktor risiko utama untuk penyakit jantung.

            7. **Platelet (platelets)**: Ini adalah jumlah platelet dalam darah, diukur dalam ribu platelet per mililiter (kiloplatelets/mL). Platelet adalah sel darah yang berperan dalam pembekuan darah. Nilai platelet dalam darah dapat memberikan informasi tentang kemampuan darah untuk membeku. Fitur ini bertipe data numerik.

            8. **Jenis Kelamin (Sex)**: Ini adalah fitur biner yang menunjukkan jenis kelamin pasien, yaitu perempuan (woman) atau laki-laki (man). Informasi ini dapat digunakan untuk mengevaluasi perbedaan jenis kelamin dalam insiden gagal jantung.

            9. **Kreatinin Serum (Serum Creatinine)**: Ini adalah tingkat kreatinin serum dalam darah, diukur dalam miligram per desiliter (mg/dL). Kreatinin adalah produk sisa metabolisme yang dapat memberikan informasi tentang fungsi ginjal. Tingkat kreatinin serum yang tinggi dapat menunjukkan masalah ginjal yang dapat mempengaruhi kondisi jantung. Fitur ini bertipe data numerik.

            10. **Natrium Serum (Serum Sodium)**: Ini adalah tingkat natrium serum dalam darah, diukur dalam miliequivalents per liter (mEq/L). Natrium adalah elektrolit penting dalam tubuh dan tingkat natrium serum dapat memberikan informasi tentang keseimbangan elektrolit yang dapat mempengaruhi fungsi jantung. Fitur ini bertipe data numerik.

            11. **Merokok (Smoking)**: Ini adalah fitur biner yang menunjukkan apakah pasien merokok atau tidak. Nilai 1 / "true" menunjukkan kebiasaan merokok, sementara 0 / "false" menunjukkan ketiadaan kebiasaan merokok. Merokok adalah faktor risiko yang signifikan dalam perkembangan penyakit jantung.

            12. **Waktu (Time)**: Ini adalah periode pemantauan atau follow-up pasien dalam satuan hari (days). Fitur ini mengukur lamanya pasien dipantau dalam penelitian. Ini adalah informasi penting dalam analisis klinis dan penelitian lanjutan. Fitur ini bertipe data numerik.

            13. **Kejadian Kematian (Death Event)**: Fitur target "death event" adalah sebuah fitur yang digunakan untuk menunjukkan apakah pasien mengalami kematian selama periode pemantauan atau tidak. Fitur ini bersifat boolean, yang berarti nilainya hanya dapat berupa dua kemungkinan: 1 / True (benar) atau 0 / False (salah).
            Jika "death event" memiliki nilai True, ini berarti bahwa pasien tersebut mengalami kematian selama periode pemantauan yang dicatat dalam dataset. Sebaliknya, jika "death event" memiliki nilai False, maka ini menunjukkan bahwa pasien tersebut masih hidup atau tidak mengalami kematian selama periode tersebut.
                 """)
    with dataset:
        url = "https://raw.githubusercontent.com/whintaaa/datapsd/main/heart_failure_clinical_records_dataset.csv"
        df = pd.read_csv(url)
        st.dataframe(df)
    with eks:
        # Baca data dari URL CSV
        url = "https://raw.githubusercontent.com/whintaaa/datapsd/main/heart_failure_clinical_records_dataset.csv"
        data = pd.read_csv(url)

        # Informasi Umum tentang Dataset
        info = data.info()

        # Statistik Deskriptif untuk Kolom-Kolom Numerik
        describe = data.describe()

        # Jumlah Nilai yang Hilang untuk Setiap Kolom
        missing_values = data.isnull().sum()

        # Beberapa Baris Pertama dari Dataset
        head = data.head()

        # Jumlah Unik untuk Kolom Target (DEATH_EVENT)
        target_counts = data['DEATH_EVENT'].value_counts()

        # Korelasi Antar Kolom Numerik
        correlation_matrix = data.corr()

        # Distribusi Umur (Age)
        age_distribution = data['age'].value_counts()

        # Visualisasi Data
        plt.figure(figsize=(15, 10))

        # Countplot untuk Kolom Target (DEATH_EVENT)
        plt.figure(figsize=(8, 5))
        sns.countplot(x='DEATH_EVENT', data=data)
        plt.title('Countplot untuk Kolom Target (DEATH_EVENT)')
        plt.xlabel('DEATH_EVENT')
        plt.ylabel('Jumlah')
        plt.show()

        st.write("\nJumlah Nilai yang Hilang:")
        st.write(missing_values)

        st.write("\nJumlah data untuk Kolom Target (DEATH_EVENT):")
        st.write(target_counts)

        # Statistik deskriptif
        statistics = data.describe()

        # Menampilkan hasil
        st.write("\nStatistik Deskriptif:")
        st.write(statistics)



with preporcessing:
    progress()
    st.title("Oversampling")
    st.write("Melihat jumlah masing-masing target pada kolom 'death event':")
    # Menghitung jumlah masing-masing target pada kolom 'death event'
    jumlah_death_event_1 = df[df['DEATH_EVENT'] == 1].shape[0]
    jumlah_death_event_0 = df[df['DEATH_EVENT'] == 0].shape[0]

    # Menampilkan jumlah masing-masing target
    st.write("Jumlah Target 'death event' dengan Nilai 1:", jumlah_death_event_1)
    st.write("Jumlah Target 'death event' dengan Nilai 0:", jumlah_death_event_0)
    st.write("""
                Bisa dilihat jumlah target dengan nilai 1 = 96 dan nilai 0 = 203 ini menandakan bahwa jumlah target pada dataset tidak seimbang. 
                Maka salah satu metode untuk menyeimbangkan target bisa menggunakan teknik oversampling.
                Teknik oversampling adalah salah satu pendekatan untuk menyeimbangkan dataset yang tidak seimbang dengan meningkatkan jumlah sampel dalam kategori minoritas. 
                Kategori minoritas adalah kelas target yang memiliki frekuensi yang lebih rendah dibandingkan dengan kelas mayoritas. Teknik oversampling dilakukan dengan cara menambahkan lebih banyak contoh dari kategori minoritas agar jumlahnya sebanding dengan kategori mayoritas.

                Ada beberapa metode oversampling yang umum digunakan, dan salah satunya adalah RandomOverSampler. Dalam RandomOverSampler, sampel acak dari kategori minoritas ditambahkan kembali ke dataset hingga jumlahnya setara dengan jumlah sampel dalam kategori mayoritas.

                Berikut adalah langkah-langkah umum untuk menggunakan teknik oversampling:

                1. Identifikasi dataset yang tidak seimbang.
                2. Pisahkan fitur (X) dan target (y).
                3. Terapkan teknik oversampling pada kategori minoritas.
                4. Gabungkan kembali data yang sudah diresampling.
                5. Lanjutkan dengan analisis atau pemodelan seperti biasa.

                Dengan menggunakan teknik oversampling, kita meningkatkan jumlah sampel di kategori minoritas (DEATH_EVENT = 1) sehingga seimbang dengan kategori mayoritas (DEATH_EVENT = 0).
            """)
    # Print the column names to identify the correct target variable
    st.write(df.columns)

    # Pilih kolom-kolom yang perlu dinormalisasi / bertype numerik
    numerical_columns = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']

    # Langkah 3: Split data menjadi fitur (X) dan target (y)
    X = df.drop(columns=['DEATH_EVENT'])
    y = df['DEATH_EVENT']

    # Menggunakan teknik oversampling dengan RandomOverSampler
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)

    # Membuat dataframe baru setelah oversampling
    df_resampled = pd.concat([X_resampled, y_resampled], axis=1)

    # Menampilkan jumlah target setelah oversampling
    st.write("Jumlah Target setelah Oversampling:")
    st.write(df_resampled['DEATH_EVENT'].value_counts())

    st.title("Normalisasi Menggunakan Zscore")
    st.write("""
                Normalisasi Z-score adalah teknik normalisasi yang digunakan untuk mengubah setiap nilai dalam suatu variabel ke dalam skala yang memiliki rata-rata nol dan deviasi standar satu. Ini adalah cara umum untuk menormalkan data sehingga nilai-nilai yang berbeda dari variabel yang sama dapat dibandingkan secara langsung.

                Proses normalisasi Z-score melibatkan mengurangkan rata-rata dari setiap nilai dalam variabel dan membaginya dengan deviasi standar. Formula normalisasi Z-score untuk suatu nilai \(x\) dalam variabel \(X\) adalah sebagai berikut:
            """)
    st.latex(r'z = \frac{{x - \text{{mean}}(X)}}{{\text{{std}}(X)}}')
    st.write("""
                di mana:
                - $( z $) adalah nilai hasil normalisasi (Z-score) dari $(x$).
                - $( \text{{mean}}(X) $) adalah rata-rata dari variabel $(X$).
                - $( \text{{std}}(X) $) adalah deviasi standar dari variabel $(X$).

                Proses ini menghasilkan distribusi data yang memiliki rata-rata nol dan deviasi standar satu. Normalisasi Z-score sangat berguna dalam beberapa konteks, terutama ketika Anda ingin membandingkan nilai-nilai dari variabel yang memiliki skala yang berbeda.

                sebelum di mormalisasi terdapat fitur boolean pada dataset maka harus dipidahkan dengan fitur numerik karena hanya fitur numerik yang akan di normalisasi kemudian normalisasi Z-score diaktifkan dengan parameter `normalize=True` dan `normalize_method='zscore'`. PyCaret akan otomatis menormalisasi fitur-fitur numerik yang ditentukan menggunakan normalisasi Z-score.
                
                Lalu secara default pycaret akan membagi dataset menjadi 70%  data train dan 30%  data test, maka untuk dataset ini 209 menjadi data train dan 90 menjadi data test.
            """)
    # Print the column names to identify the correct target variable
    st.write(df.columns)
    # Pilih kolom-kolom yang perlu dinormalisasi / bertype numerik
    numerical_columns = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']

    # Split data menjadi fitur (X) dan target (y)
    X = df.drop(columns=['DEATH_EVENT'])
    y = df['DEATH_EVENT']

    # Inisialisasi ekperimen PyCaret
    exp = setup(data=df, target='DEATH_EVENT', normalize=True, normalize_method='zscore', numeric_features=numerical_columns)
    
    local_image_path = "img/norm.png"
    local_image = Image.open(local_image_path)
    st.image(local_image, caption="Random Forest Classifier", use_column_width=True)
with modeling:
    progress()
    st.title("Mencari Model Terbaik Menggunakan Pycaret")
    st.write("""
                Pycaret adalah library Python yang menyederhanakan proses pengembangan model machine learning. 
                Dengan fitur otomatis seperti setup data, pemilihan model, optimasi hyperparameter, dan visualisasi hasil, Pycaret memungkinkan pengguna untuk fokus pada inti pemodelan tanpa menulis banyak kode. 
                Berikut implementasi pycaret untuk mencari model terbaik untuk memprediksi resiko gagal jantung pada pasien:
            """)
    # Bandingkan model dan cari yang terbaik
    # best_model = compare_models()
    # st.write(best_model)
    local_image_path = "img/model.png"
    local_image = Image.open(local_image_path)
    st.image(local_image, caption="Model terbaik", use_column_width=True)

    st.title("Menyimpan Model terbaik Menggunakan Pycaret")
    st.write("""
                Bisa dilihat diatas bahwa model terbaik salah satunya adalah Random Forest Classifier dengan akurasi 0.8517. maka kita simpan model Random Forest Classifier untuk prediksi nantinya.

                #### Random Forest Classifier:
                Random Forest adalah algoritma machine learning yang termasuk dalam kategori ensemble learning. Ensemble learning menggabungkan prediksi dari beberapa model untuk meningkatkan performa dan ketahanan terhadap overfitting. Random Forest dapat digunakan untuk tugas klasifikasi (seperti prediksi kategori) dan regresi (prediksi nilai numerik).

                #### Cara Kerja:

                1. **Pembuatan Banyak Pohon (Trees):**
                - Random Forest terdiri dari sejumlah besar pohon keputusan yang dibuat secara acak. Setiap pohon dalam Random Forest dibuat berdasarkan subset acak dari data pelatihan dan fitur-fiturnya.

                2. **Bootstrap Sampling (Bootstrapped Dataset):**
                - Pada setiap langkah pembuatan pohon, dilakukan bootstrap sampling, yaitu pengambilan sampel acak dengan penggantian dari dataset pelatihan. Beberapa data dapat muncul lebih dari sekali, dan beberapa mungkin tidak dipilih.

                3. **Pemilihan Fitur Secara Acak:**
                - Pada setiap langkah pembuatan pohon, juga dilakukan pemilihan acak dari fitur-fitur yang tersedia. Ini membantu dalam menciptakan variasi antar pohon.

                4. **Pembuatan Pohon Keputusan:**
                - Setiap pohon dibuat menggunakan data sampel dari bootstrap dan fitur yang dipilih secara acak. Pemisahan (split) di setiap node pohon dilakukan berdasarkan kriteria seperti Gini Impurity untuk klasifikasi atau Mean Squared Error untuk regresi.

                5. **Voting (Klasifikasi):**
                - Untuk tugas klasifikasi, setelah semua pohon selesai membuat prediksi, hasilnya diambil berdasarkan mayoritas voting. Kelas dengan voting terbanyak dianggap sebagai prediksi akhir.
                jika \(N\) adalah jumlah pohon dalam Random Forest, dan \(h_i(x)\) adalah hasil prediksi dari pohon ke-i, maka hasil akhir \(H(x)\) dari Random Forest dapat dihitung sebagai berikut:

                **Rumus untuk Klasifikasi Random Forest:**
            """)
    st.latex(r'H(x) = \text{{mode}}(h_1(x), h_2(x), \ldots, h_N(x))')

    st.write("""
            Rumus ini, sesuai dengan prinsip mayoritas voting pada ensambel model seperti Random Forest. Model ensambel, seperti Random Forest, cenderung memberikan performa yang baik dalam berbagai jenis dataset, termasuk dataset kesehatan seperti "Heart failure clinical records".

            Dalam kasus dataset kesehatan seperti ini, Random Forest bisa menjadi pilihan yang baik karena:
            
            1. Robust terhadap Overfitting: Random Forest mampu mengatasi masalah overfitting yang mungkin muncul pada pohon keputusan tunggal, karena hasil mayoritas dari banyak pohon keputusan.
            
            2. Tidak Sensitif terhadap Outliers: Random Forest dapat menangani data yang tidak seimbang dan keberadaan outlier dalam dataset.
            
            3. Interpretability: Meskipun Random Forest cenderung tidak seinterpretatif pohon keputusan tunggal, tetapi masih memberikan pemahaman yang baik tentang pentingnya fitur dalam membuat keputusan.
            
            4. Handling Fitur Numerik dan Kategorikal: Random Forest dapat menangani baik fitur numerik maupun kategorikal tanpa memerlukan transformasi khusus.
            
            5. Performa yang Baik secara Umum: Random Forest umumnya memberikan performa yang baik tanpa perlu penyesuaian parameter yang terlalu rumit.

            """)
    
    # best_model = create_model('rf')

    # Simpan model terbaik ke dalam file pickle
    model_filename = 'best_model.pkl'
    # with open(model_filename, 'wb') as file:
    #   pickle.dump(best_model, file)

    # Load model dari file pickle
    with open(model_filename, 'rb') as file:
        loaded_model = pickle.load(file)

    local_image_path = "img/model_rf.png"
    local_image = Image.open(local_image_path)
    st.image(local_image, caption="Random Forest Classifier", use_column_width=True)

with implementation:
    st.title("Prediksi Death_Event Menggunakan Random Forest Classifier (rf) dengan Data Baru")
    st.write("""
                            
                ### Prediksi Menggunakan Random Forest untuk Klasifikasi:

                1. **Persiapkan Data Baru:**
                - Masukkan data baru ke dalam model dengan memberikan nilai untuk setiap fitur yang sesuai.

                2. **Lakukan Prediksi pada Setiap Pohon:**
                - Setiap pohon dalam Random Forest memberikan prediksi berdasarkan data baru.

                3. **Aggregasi Hasil Prediksi:**
                - Hasil prediksi dari setiap pohon diambil dan dihitung mayoritas voting.

                4. **Tentukan Kelas Akhir:**
                - Hasil akhir diambil berdasarkan mayoritas voting sebagai kelas prediksi akhir.

                ### Contoh Kode:
                ```python
                # Membaca data baru yang akan diprediksi
                new_data = pd.DataFrame([[age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time]], columns=X.columns)

                # Memuat model terbaik dari file pickle
                loaded_model = load_model('best_model')

                # Melakukan prediksi pada data baru
                predictions = predict_model(loaded_model, data=new_data)

                # Menampilkan hasil prediksi
                if predictions['prediction_label'].iloc[0] == 1:
                    print("Hasil Prediksi: Pasien berisiko mengalami DEATH_EVENT")
                else:
                    print("Hasil Prediksi: Pasien tidak berisiko mengalami DEATH_EVENT")
                ```

                Dalam kode ini, `prediction_label` adalah kolom yang berisi prediksi kelas (0 atau 1) dari model Random Forest untuk tugas klasifikasi. Jika nilai Label adalah 1, itu berarti pasien berisiko mengalami DEATH_EVENT; jika 0, itu berarti pasien tidak berisiko.
            """)

    st.write("Masukkan nilai-nilai fitur untuk memprediksi DEATH_EVENT:")
    # Simpan model terbaik ke dalam file pickle
    model_filename = 'best_model.pkl'
    # Load model dari file pickle
    with open(model_filename, 'rb') as file:
        loaded_model = pickle.load(file)


    # Minta pengguna untuk memasukkan nilai-nilai fitur
    age = st.number_input("Age", min_value=0.0)
    anaemia = st.radio("Anaemia (0 untuk Tidak, 1 untuk Ya)", [0, 1])
    creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase", min_value=0.0)
    diabetes = st.radio("Diabetes (0 untuk Tidak, 1 untuk Ya)", [0, 1])
    ejection_fraction = st.number_input("Ejection Fraction", min_value=0.0)
    high_blood_pressure = st.radio("High Blood Pressure (0 untuk Tidak, 1 untuk Ya)", [0, 1])
    platelets = st.number_input("Platelets", min_value=0.0)
    serum_creatinine = st.number_input("Serum Creatinine", min_value=0.0)
    serum_sodium = st.number_input("Serum Sodium", min_value=0.0)
    sex = st.radio("Sex (0 untuk Perempuan, 1 untuk Laki-laki)", [0, 1])
    smoking = st.radio("Smoking (0 untuk Tidak, 1 untuk Ya)", [0, 1])
    time = st.number_input("Time", min_value=0.0)

    if st.button("Prediksi DEATH_EVENT"):
        # Membuat data baru untuk prediksi
        new_data = pd.DataFrame([[age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time]], columns=X.columns)

        # Melakukan prediksi pada data baru
        predictions = predict_model(loaded_model, data=new_data)

        # Hasil prediksi
        st.write("Hasil Prediksi:")
        if predictions['prediction_label'][0] == 1:
            st.write("Pasien berisiko mengalami DEATH_EVENT")
        else:
            st.write("Pasien tidak berisiko mengalami DEATH_EVENT")

