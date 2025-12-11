# 🍃 Spring Boot AI Assistant (Llama 3.1 + RAG + Fine-Tuning) on Colab T4

Bu proje, **Google Colab (Free Tier)** üzerinde sunulan **Tesla T4 GPU** ile çalışacak şekilde özel olarak optimize edilmiştir.

**Unsloth** kütüphanesi sayesinde, normalde çok daha güçlü donanım gerektiren **Llama 3.1 8B** modeli, Colab'ın ücretsiz T4 GPU'sunda hem eğitilebilir (Fine-Tuning) hem de RAG (Retrieval-Augmented Generation) sistemiyle birleştirilerek çalıştırılabilir.

---

## 🚀 Özellikler

* **⚡ Colab T4 Optimizasyonu:** Unsloth teknolojisi ile model, T4 GPU'nun 16GB VRAM'ine sığacak şekilde (4-bit quantization) sıkıştırılmıştır.
* **🧠 Fine-Tuning (SFT):** Model, Spring Boot kodlama standartlarını öğrenmek için çalışma anında eğitilir.
* **📚 RAG Mimarisi:** Güncel Spring Boot dokümantasyonunu vektör veritabanında (FAISS) tutar ve halüsinasyonu önler.
* **translator-agent:** Türkçe sorulan teknik soruları, vektör arama başarısını artırmak için arka planda İngilizceye çevirir.
* **🌐 Web Arayüzü:** Colab içinde çalışan sistemi, dış dünyaya açan modern bir Chat arayüzü sunar.
* **🔗 Ngrok Tünelleme:** Colab'ın local portunu internete açarak tarayıcıdan erişim sağlar.

---

## 🛠️ Kurulum ve Ortam Hazırlığı

Bu projeyi çalıştırmak için Google Colab üzerinde GPU hızlandırmayı aktif etmeniz yeterlidir.

### 1. Colab Ayarları
Colab menüsünden şu adımları izleyin:
1.  **Çalışma Zamanı (Runtime)** > **Çalışma Zamanı Türünü Değiştir (Change runtime type)**
2.  **Donanım Hızlandırıcı (Hardware accelerator):** `T4 GPU` seçin.
3.  **Kaydet** diyerek onaylayın.

### 2. Gerekli Dosyalar
Kodun hatasız çalışması için sol menüdeki **Dosyalar** kısmına şu iki dosyayı sürükleyip bırakın:
* `spring_boot_finetune_full.jsonl`: Fine-tuning için soru-cevap çiftleri.
* `spring_boot_rag_llamaparse.json`: RAG sistemi için ham dokümantasyon verisi.

---

## 🧩 Sistem Mimarisi ve Çalışma Mantığı

Sistem, T4 GPU sınırları içinde kalacak şekilde 5 aşamalı bir mimariye sahiptir:

### 1. Model Yükleme (4-bit Quantization)
`unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` modeli yüklenir.
* **Neden 4-bit?** Normalde 8 milyar parametreli bir model yaklaşık 16GB+ VRAM gerektirir. 4-bit yükleme ile bu gereksinim düşürülerek T4 GPU üzerinde çalışması sağlanır.

### 2. LoRA Adaptörleri (PEFT)
Modelin tamamı eğitilmez (bu T4'ü çökertirdi). Bunun yerine **LoRA (Low-Rank Adaptation)** tekniği ile modelin sadece %1-2'lik kısmına "yama" yapılır.
* **Target Modules:** `["q_proj", "k_proj", ...]` Modelin dikkat mekanizmaları eğitilerek Spring Boot bilgisi aşılanır.

### 3. Veri Temizleme ve RAG Hazırlığı
Yüklenen ham JSON verisi, kod içindeki temizleyici (cleaner) script ile işlenir:
* Gürültülü veriler (İçindekiler, Lisanslar) silinir.
* Veriler `sentence-transformers` ile vektörlere çevrilip RAM üzerinde çalışan **FAISS** veritabanına kaydedilir.

### 4. Fine-Tuning (Eğitim)
Model, yüklediğiniz `jsonl` verisi ile hızlı bir eğitime tabi tutulur.
* **Süre:** Yaklaşık 5-10 dakika (Veri boyutuna göre değişir).
* **Sonuç:** Model artık Spring Boot 3.0+ kodlama standartlarına daha aşina hale gelir.

### 5. Flask API ve Chat Akışı
Kullanıcı arayüzü üzerinden gelen sorular şu akışla işlenir:
1.  **Çeviri:** Soru arka planda İngilizceye çevrilir (Daha iyi doküman bulmak için).
2.  **RAG:** FAISS veritabanından en alakalı 3 kod parçası çekilir.
3.  **Üretim:** Bulunan parçalar ve kullanıcının Türkçe sorusu modele verilir.
4.  **Cevap:** Model, RAG bilgisini kullanarak Spring Boot 3.2 uyumlu cevap üretir.

---

## ⚙️ Önemli Parametreler

Kod içerisindeki bu parametreler T4 GPU performansı için kritiktir:

| Parametre | Değer | Açıklama |
| :--- | :--- | :--- |
| `MAX_SEQ_LENGTH` | `2048` | T4 belleğini aşmamak için güvenli sınırdır. Daha fazla artırılırsa "Out of Memory" hatası alınabilir. |
| `load_in_4bit` | `True` | T4 GPU'da çalışabilmesi için zorunludur. |
| `per_device_train_batch_size` | `2` | Eğitim sırasında aynı anda işlenen veri sayısı. T4 için 2 idealdir. |
| `gradient_accumulation_steps` | `4` | Küçük batch size açığını kapatmak için gradyanlar biriktirilir. |
| `temperature` | `0.3` | Modelin yaratıcılık seviyesi. Kod üretimi için düşük tutulmuştur. |

---

## 🖥️ Adım Adım Çalıştırma Kılavuzu

1.  **Token Girin:**
    Kodun 2. hücresindeki `NGROK_AUTH_TOKEN` alanına Ngrok'tan aldığınız ücretsiz token'ı yapıştırın.

2.  **Hücreleri Çalıştırın:**
    Yukarıdan aşağıya doğru `Play` (▶) butonlarına basarak ilerleyin.
    * *Hücre 1:* Gerekli kütüphaneleri kurar (Unsloth, LangChain vb.).
    * *Hücre 2:* Modeli T4 GPU'ya yükler.
    * *Hücre 3:* Modeli eğiter (Fine-Tune).
    * *Hücre 4:* Dokümanları temizler ve veritabanını kurar.
    * *Hücre 5:* Web sunucusunu başlatır.

3.  **Sisteme Erişim:**
    En son hücre çalıştığında ekrana gelen `👉 Arayüze Gitmek İçin Tıkla: https://....ngrok-free.app` linkine tıklayın.

---

## ⚠️ İpuçları

* **Oturum Süresi:** Google Colab ücretsiz sürümü tarayıcı sekmesi kapandığında veya belirli bir süre işlem yapılmadığında oturumu sonlandırabilir.
* **İlk Yanıt:** Model ilk soruda "soğuk başlangıç" nedeniyle 10-20 saniye bekletebilir, sonraki yanıtlar hızlanacaktır.
* **Disk Alanı:** Unsloth kütüphanesi disk alanını verimli kullanır ancak Drive bağlantısı yaparsanız modelleri oraya da yedekleyebilirsiniz.

---

## 📝 Lisans
Bu proje eğitim ve deneme amaçlıdır. Llama 3.1 modeli Meta lisansına tabidir.
