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

## 📝 Lisans ve Feragatname
Bu proje eğitim ve araştırma amaçlı geliştirilmiştir. Kullanılan `Llama 3.1` modeli Meta lisans koşullarına, `Spring Boot` markası VMware lisans haklarına tabidir.
