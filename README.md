# 🍃 Spring Boot AI Assistant (Hybrid: RAG + Fine-Tuning)

This project is an advanced AI assistant specialized for the **Spring Boot** ecosystem, combining **RAG (Retrieval-Augmented Generation)** and **Fine-Tuning** techniques.

Built to run on **Google Colab T4 GPU** using the **Unsloth** optimization framework, it utilizes the **Meta-Llama-3.1-8B-Instruct** model. It delivers both precise information retrieval based on documentation (RAG) and internalized domain expertise (Fine-Tuning).

---

## 🚀 About the Project & Development Process

Unlike standard chatbots, this system is built on a custom data pipeline derived from raw documentation processing:

### 1. 📄 Data Processing (Llama Parse)
The development process began by parsing the official **Spring Boot PDF documentation**. The **Llama Parse** library was used to transform complex PDF structures into machine-readable, structured text suitable for processing.

### 2. 🧠 Dataset Generation (OpenAI API)
To transform the model from a simple "reader" into an "expert," the parsed documents were processed via the **OpenAI API**. High-quality **Question-Answer pairs** were generated to create a structured JSON **Fine-Tuning dataset**.

### 3. 🎯 Hybrid Architecture (RAG + FT)
* **RAG (Knowledge Base):** The parsed content was converted into a vector-based JSON knowledge base to allow the model to access real-time, up-to-date information.
* **Fine-Tuning (Expertise):** Using the generated Q&A pairs, the Llama 3.1 8B model was fine-tuned on **Colab T4** to deeply understand Spring Boot concepts and coding standards.

---

## ⚡ Features

* **T4 GPU Optimization:** Thanks to Unsloth and 4-bit quantization, the entire system runs smoothly on the free Colab T4 GPU.
* **Smart Translation Agent:** Automatically translates non-English queries into English in the background to improve RAG retrieval accuracy while preserving technical terminology.
* **Web Interface:** Features a modern, ChatGPT-like interface with syntax highlighting.
* **Public Access:** Exposes the local Colab server to the internet via Ngrok tunneling.

---

## 🔮 Future Plans (Roadmap)

The project is currently under active development. Future goals include:
* 📚 Integrating more extensive official documentation.
* 💻 Adding real-world scenarios and code patterns from GitHub repositories to the dataset.
* 📈 Increasing the dataset volume and further improving the model's accuracy.

---

## 🛠️ Installation & Usage on Colab

### Required Files
Upload the following two files to the file manager in the left sidebar:
1.  `spring_boot_finetune_full.jsonl`: The Q&A training set generated via OpenAI.
2.  `spring_boot_rag_llamaparse.json`: The RAG data source parsed via Llama Parse.

### Step-by-Step Execution
1.  **Token Setup:** Paste your **Ngrok Auth Token** into the variable in Cell 2.
2.  **Sequential Execution:** Run the cells from top to bottom:
    * *Setup -> Load Model -> Fine-Tune -> RAG Prep -> Web Server*
3.  **Access:** Click the public link (`https://....ngrok-free.app`) generated in the final cell output.

---

## ⚙️ Technical Parameters

Critical settings used to ensure stability on the T4 GPU:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `Model` | `unsloth/Meta-Llama-3.1-8B...` | Compressed version using 4-bit Quantization. |
| `MAX_SEQ_LENGTH` | `2048` | Token limit set to manage T4 VRAM usage. |
| `LoRA Rank (r)` | `16` | Parameter density trained during fine-tuning. |
| `Temperature` | `0.3` | Low creativity setting to prevent hallucinations in code generation. |

---



# 🍃 Spring Boot AI Assistant (Hybrid: RAG + Fine-Tuning)

Bu proje, **Spring Boot** ekosistemi için özelleştirilmiş, **RAG (Retrieval-Augmented Generation)** ve **Fine-Tuning (İnce Ayar)** tekniklerini birleştiren ileri seviye bir Yapay Zeka asistanıdır.

**Google Colab T4 GPU** üzerinde **Unsloth** optimizasyon çatısı kullanılarak geliştirilen sistem, **Meta-Llama-3.1-8B-Instruct** modelini temel alır. Hem dokümantasyona dayalı kesin bilgi erişimi (RAG) hem de modelin içselleştirilmiş bilgi yeteneğini (Fine-Tuning) bir arada sunar.

---

## 🚀 Proje Hakkında ve Geliştirme Süreci

Bu sistem sıradan bir chatbot uygulamasından farklı olarak, ham dokümantasyonun işlenmesiyle oluşturulmuş özel bir veri hattı (pipeline) üzerine kuruludur:

### 1. 📄 Veri İşleme (Llama Parse)
Geliştirme süreci, resmi **Spring Boot PDF dokümantasyonunun** işlenmesiyle başladı. Karmaşık PDF yapısını anlamlı metinlere dönüştürmek için **Llama Parse** kütüphanesi kullanıldı. Bu işlem sonucunda ham veri, makine tarafından okunabilir yapısal bir formata dönüştürüldü.

### 2. 🧠 Veri Seti Üretimi (OpenAI API)
Modelin sadece "okuyan" değil, "anlayan" bir uzmana dönüşmesi için ayrıştırılan dokümanlar **OpenAI API** ile işlendi. Bu aşamada, yüksek kaliteli **Soru-Cevap (Question-Answer)** çiftleri üretilerek JSON formatında bir **Fine-Tuning veri seti** oluşturuldu.

### 3. 🎯 Hibrit Mimari (RAG + FT)
* **RAG (Bilgi Bankası):** Ayrıştırılan içerikler, modelin anlık ve güncel bilgiye erişebilmesi için vektör tabanlı bir JSON bilgi bankasına dönüştürüldü.
* **Fine-Tuning (Uzmanlık):** Üretilen soru-cevap setleri ile Llama 3.1 8B modeli **Colab T4** üzerinde eğitilerek, Spring Boot konseptlerine ve kodlama tarzına hakim olması sağlandı.

---

## ⚡ Özellikler

* **T4 GPU Optimizasyonu:** Unsloth ve 4-bit quantization sayesinde tüm sistem ücretsiz Colab GPU'sunda çalışır.
* **Akıllı Çeviri Ajanı:** Türkçe soruları arka planda teknik terminolojiye sadık kalarak İngilizceye çevirir ve RAG başarısını artırır.
* **Web Arayüzü:** Syntax highlighting destekli, ChatGPT benzeri modern bir arayüz.
* **Dışa Açılım:** Ngrok tünellemesi ile yerel sunucuyu internete açar.

---

## 🔮 Gelecek Planları (Roadmap)

Proje şu anda aktif geliştirme aşamasındadır. İlerleyen dönemler için hedeflenen geliştirmeler:
* 📚 Daha fazla resmi dokümantasyonun entegrasyonu.
* 💻 Gerçek dünya senaryolarını kapsayan GitHub repolarının veri setine eklenmesi.
* 📈 Veri setinin hacminin artırılması ve modelin doğruluk oranının iyileştirilmesi.

---

## 🛠️ Kurulum ve Colab Kullanımı

### Gerekli Dosyalar
Sol menüdeki dosya yöneticisine şu iki dosyayı yükleyin:
1.  `spring_boot_finetune_full.jsonl`: OpenAI ile üretilmiş Soru-Cevap eğitim seti.
2.  `spring_boot_rag_llamaparse.json`: Llama Parse ile ayrıştırılmış RAG veri kaynağı.

### Adım Adım Çalıştırma
1.  **Token Ayarı:** Kodun 2. hücresine **Ngrok Auth Token**'ınızı yapıştırın.
2.  **Sıralı Başlatma:** Hücreleri yukarıdan aşağıya sırasıyla çalıştırın.
    * *Kurulum -> Model Yükleme -> Eğitim (Fine-Tune) -> RAG Hazırlığı -> Web Sunucusu*
3.  **Erişim:** Son hücredeki `https://....ngrok-free.app` linkine tıklayın.

---

## ⚙️ Teknik Parametreler

Sistemin T4 GPU üzerinde stabil çalışması için kullanılan kritik ayarlar:

| Parametre | Değer | Açıklama |
| :--- | :--- | :--- |
| `Model` | `unsloth/Meta-Llama-3.1-8B...` | 4-bit Quantization ile sıkıştırılmış versiyon. |
| `MAX_SEQ_LENGTH` | `2048` | T4 belleğini yönetmek için belirlenen token sınırı. |
| `LoRA Rank (r)` | `16` | Fine-tuning sırasında eğitilen parametre yoğunluğu. |
| `Temperature` | `0.3` | Kod üretiminde halüsinasyonu önlemek için düşük yaratıcılık ayarı. |

---





