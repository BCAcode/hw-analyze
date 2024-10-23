import streamlit as st
import cv2
import pytesseract
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF
import nltk
from nltk.corpus import stopwords
from io import BytesIO

# NLTK stopwords indir (ilk çalıştırmada sadece bir kez gerekli)
nltk.download('stopwords')

# Başlık
st.title("Ödev Benzerlik Analizi")

# Kullanıcıdan dosya yüklemesini iste
uploaded_files = st.file_uploader("Lütfen ödev dosyalarını yükleyin", type=["txt", "png", "jpg"], accept_multiple_files=True)

if uploaded_files:
    texts = []
    filenames = []

    # Her bir dosya için işlem yap
    for uploaded_file in uploaded_files:
        filenames.append(uploaded_file.name)
        
        if uploaded_file.type == "text/plain":
            # Metin dosyası
            text = uploaded_file.read().decode("utf-8")
            texts.append(text)
        else:
            # Görüntü dosyası için OCR uygulaması
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            text = pytesseract.image_to_string(image)
            texts.append(text)

    if st.button("Analizi Başlat"):
        # Benzerlik hesaplama
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Sonuçları göster
        st.write("Benzerlik Oranları:")
        results = []
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                if similarity_matrix[i][j] > 0.5:  # %50'den fazla benzerlik
                    # Benzerlik türünü belirle
                    similarity_type = ""
                    if uploaded_files[i].type == uploaded_files[j].type:
                        if texts[i].strip() == texts[j].strip():
                            similarity_type = "Aynı ödev, aynı görsel"
                        else:
                            similarity_type = "Aynı ödev, farklı görsel"
                    else:
                        similarity_type = "Farklı ödev, benzer metin"
                    
                    results.append((filenames[i], filenames[j], similarity_matrix[i][j], similarity_type))

        if results:
            for file1, file2, score, sim_type in results:
                st.write(f"{file1} ve {file2} arasında benzerlik: {score:.2f} - Tür: {sim_type}")

            # PDF raporu oluşturma
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.set_font("Arial", size=12)

            # Sayfaları iki görsel olarak düzenle
            for i in range(0, len(results), 2):
                pdf.add_page()
                if i < len(results):
                    file1, file2, score, sim_type = results[i]
                    pdf.cell(200, 10, txt=f"{file1} ve {file2} arasında benzerlik: {score:.2f} - Tür: {sim_type}", ln=True)
                    # Görsel ekleme
                    pdf.image(filenames[uploaded_files.index(uploaded_files[i])], x=10, y=30, w=90)  # İlk görsel
                    pdf.ln(60)  # Görselin altında boşluk bırak
                    pdf.cell(200, 10, txt=f"Ödev: {file1}", ln=True)
                
                if i + 1 < len(results):
                    file1, file2, score, sim_type = results[i + 1]
                    pdf.cell(200, 10, txt=f"{file1} ve {file2} arasında benzerlik: {score:.2f} - Tür: {sim_type}", ln=True)
                    # Görsel ekleme
                    pdf.image(filenames[uploaded_files.index(uploaded_files[i + 1])], x=110, y=30, w=90)  # İkinci görsel
                    pdf.ln(60)  # Görselin altında boşluk bırak
                    pdf.cell(200, 10, txt=f"Ödev: {file1}", ln=True)

            # PDF dosyasını kaydet
            pdf_file_path = "benzerlik_raporu.pdf"
            pdf.output(pdf_file_path)

            # PDF'yi indir butonu
            with open(pdf_file_path, "rb") as pdf_file:
                st.download_button(label="PDF Olarak İndir", data=pdf_file, file_name=pdf_file_path, mime='application/pdf')
        else:
            st.write("Benzerlik bulunamadı.")
