# ==============================================================================
# HÜCRE 1: KURULUM VE IMPORTLAR
# ==============================================================================
print("⏳ Kütüphaneler kuruluyor... (Bu işlem 1-2 dakika sürebilir)")
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" -q
!pip install --no-deps xformers trl peft accelerate bitsandbytes -q
!pip install langchain langchain-community faiss-cpu sentence-transformers -q
!pip install flask pyngrok flask-cors -q
!pip install langchain-huggingface -q

import os
import json
import torch
from flask import Flask, request, jsonify
from pyngrok import ngrok
from flask_cors import CORS

# RAG ve Model Kütüphaneleri
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset

print("✅ Kurulum tamamlandı. Sonraki hücreye geçebilirsiniz.")

# ==============================================================================
# HÜCRE 2: MODEL YÜKLEME VE AYARLAR
# ==============================================================================

# --- AYARLAR ---
NGROK_AUTH_TOKEN = "SECRET-TOKEN"
NGROK_STATIC_DOMAIN = "SECRET-URL"
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048

# Ngrok Yetkilendirme
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

print("\n🚀 Model GPU'ya yükleniyor...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = None,
    load_in_4bit = True,
)

# LoRA Adaptörlerini ekle
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

print("✅ Model Yüklendi ve Hazır.")

# ==============================================================================
# HÜCRE 3: FINE-TUNING (EĞİTİM)
# ==============================================================================

# Prompt Formatı
alpaca_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Sen bir Spring Boot ve Java uzmanısın. Kullanıcının teknik sorularına, dokümantasyona dayalı, en iyi uygulama (best practice) standartlarına uygun cevaplar ver.<|eot_id|><|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{}<|eot_id|>"""

# Veri Temizleme Fonksiyonu
def load_and_sanitize_data(file_path):
    safe_data = []
    print(f"🛠️ Veri seti temizlenerek yükleniyor: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    item = json.loads(line)
                    safe_item = {
                        "instruction": str(item.get("instruction", "")),
                        "input": str(item.get("input", "")),
                        "output": str(item.get("output", ""))
                    }
                    if safe_item["instruction"] and safe_item["output"]:
                        safe_data.append(safe_item)
                except: continue
        print(f"✅ {len(safe_data)} satır veri başarıyla yüklendi.")
        return Dataset.from_list(safe_data)
    except FileNotFoundError:
        print("❌ HATA: Finetune dosyası bulunamadı!")
        return None

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs      = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        texts.append(alpaca_prompt.format(instruction, output) + tokenizer.eos_token)
    return { "text" : texts, }

# Eğitimi Başlat
try:
    dataset = load_and_sanitize_data("spring_boot_finetune_full.jsonl")

    if dataset:
        dataset = dataset.map(formatting_prompts_func, batched = True)

        print("🧠 Eğitim Başlıyor...")
        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = dataset,
            dataset_text_field = "text",
            max_seq_length = MAX_SEQ_LENGTH,
            dataset_num_proc = 2,
            packing = False,
            args = TrainingArguments(
                per_device_train_batch_size = 2,
                gradient_accumulation_steps = 4,
                warmup_steps = 5,
                max_steps = 400,
                learning_rate = 2e-4,
                fp16 = not torch.cuda.is_bf16_supported(),
                bf16 = torch.cuda.is_bf16_supported(),
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = 3407,
                output_dir = "outputs",
            ),
        )
        trainer.train()
        print("✅ Fine-Tuning Tamamlandı! Model hafızada güncellendi.")
    else:
        print("⚠️ Veri seti yok, Base Model ile devam ediliyor.")

except Exception as e:
    print(f"⚠️ Eğitim Hatası: {e}")


# ==============================================================================
# RAG DATA CLEANER & OPTIMIZER (VERİ TEMİZLEME ROBOTU)
# ==============================================================================
import json
import re

# Dosya isimleri
input_file = "spring_boot_rag_llamaparse.json"
output_file = "spring_boot_rag_OPTIMIZED.json"

print("🔄 RAG Verisi Optimize Ediliyor...")

try:
    with open(input_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
except FileNotFoundError:
    print("❌ HATA: JSON dosyası bulunamadı! Lütfen dosya adını kontrol et.")
    raw_data = []

cleaned_data = []
current_chapter = "General"
buffer_text = ""

# --- TEMİZLİK KURALLARI (REGEX) ---
# 1. Versiyon Listeleri (spring-security.version vb.)
version_pattern = re.compile(r'-\s[\w\-\.]+\.version')
# 2. İçindekiler Tablosu
toc_pattern = re.compile(r'Table of Contents', re.IGNORECASE)
# 3. Yasal Uyarılar / Yazarlar
legal_pattern = re.compile(r'(Phillip Webb|Dave Syer|Apache License)', re.IGNORECASE)

total_chunks = len(raw_data)
kept_chunks = 0

for chunk in raw_data:
    content = chunk.get("content", "")

    # --- ADIM 1: GÜRÜLTÜ FİLTRESİ ---

    # A) İçindekiler sayfasını atla
    if toc_pattern.search(content):
        continue

    # B) Sürüm listelerini (Dependency Versions) temizle
    if len(version_pattern.findall(content)) > 5: # 5'ten fazla versiyon satırı varsa çöptür
        continue

    # C) Yazar listesi ve Lisansları atla
    if legal_pattern.search(content):
        continue

    # D) Çok kısa (boş) sayfaları atla
    if len(content) < 50:
        continue

    # --- ADIM 2: BAĞLAM YAKALAMA (Context Injection) ---
    # Eğer satır "# Chapter" ile başlıyorsa, o başlığı hafızaya al
    lines = content.split('\n')
    for line in lines[:3]:
        if line.strip().startswith("# ") and len(line) < 100:
            current_chapter = line.strip().replace("#", "").strip()
            break

    # --- ADIM 3: ZENGİNLEŞTİRME ---
    # Her parçanın başına "Bu parça şu bölümden geliyor" diye etiket yapıştırıyoruz.
    # Böylece model "Service" ararken "Giriş" bölümündeki kodla karıştıramaz.
    enriched_content = f"CONTEXT: Spring Boot Reference - Section: {current_chapter}\n\n{content}"

    # --- ADIM 4: BİRLEŞTİRME (Merging) ---
    # 300 karakterden kısa parçaları tek başına alma, bir sonrakine ekle (Buffer)
    if len(content) < 300:
        buffer_text += "\n\n" + enriched_content
        continue
    else:
        if buffer_text:
            enriched_content = buffer_text + "\n\n" + enriched_content
            buffer_text = ""

    # Temizlenmiş veriyi kaydet
    new_chunk = {
        "content": enriched_content,
        "metadata": chunk.get("metadata", {})
    }
    cleaned_data.append(new_chunk)
    kept_chunks += 1

# Kalan buffer'ı ekle
if buffer_text:
    cleaned_data.append({"content": buffer_text, "metadata": {"source": "merged"}})

# Dosyayı Kaydet
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

print(f"✅ İŞLEM TAMAMLANDI!")
print(f"   - Ham Parça Sayısı: {total_chunks}")
print(f"   - Temizlenmiş Parça: {len(cleaned_data)}")
print(f"   - Yeni Dosya: {output_file}")
print("👉 Şimdi HÜCRE 4'teki kodda dosya adını 'spring_boot_rag_OPTIMIZED.json' olarak değiştir.")



# ==============================================================================
# HÜCRE 4: RAG SİSTEMİ (Vektör Veritabanı)
# ==============================================================================
print("\n📚 RAG Sistemi Kuruluyor...")

try:
    with open("spring_boot_rag_OPTIMIZED.json", "r", encoding="utf-8") as f:
        rag_data = json.load(f)

    # Gürültü Filtresi
    BLACKLIST_KEYWORDS = [
        "Table of Contents", "Phillip Webb", "1. Legal", "2. Getting Help",
        "Documentation Overview", "Upgrading From an Earlier Version"
    ]

    documents = []
    skipped_count = 0

    for chunk in rag_data:
        content = chunk.get("content", "")
        if len(content) < 100: continue

        is_noise = False
        for keyword in BLACKLIST_KEYWORDS:
            if keyword in content[:200]:
                is_noise = True
                break

        if is_noise:
            skipped_count += 1
            continue

        doc = Document(page_content=content, metadata={"source": "Spring Boot Docs"})
        documents.append(doc)

    print(f"🧹 {skipped_count} adet gürültülü parça temizlendi.")

    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(documents, embed_model)
    print(f"✅ Vektör Veritabanı Hazır! {len(documents)} kaliteli parça indekslendi.")

except Exception as e:
    print(f"❌ RAG Hatası: {e}")
    vector_db = None


# ==============================================================================
# HÜCRE 5: FLASK API, UI & NGROK (TAM KOD)
# ==============================================================================
print("\n🌐 Web Sunucusu ve Arayüz Hazırlanıyor...")

from flask import Flask, request, jsonify, render_template_string
from pyngrok import ngrok
from flask_cors import CORS
import torch

# --- 1. MODERN CHATGPT BENZERİ ARAYÜZ (HTML/CSS/JS) ---
# Bu HTML kodunu Python stringi içine gömüyoruz ki tek dosya çalışsın.
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Spring Boot AI Asistanı</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/java.min.js"></script>

    <style>
        body { background-color: #343541; color: #ececf1; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .chat-container { max-width: 800px; margin: 0 auto; height: 90vh; display: flex; flex-direction: column; }
        .messages { flex-grow: 1; overflow-y: auto; padding: 20px; scrollbar-width: thin; scrollbar-color: #555 #343541; }

        /* Mesaj Baloncukları */
        .message { display: flex; padding: 20px; border-bottom: 1px solid #2d2d3a; }
        .message.user { background-color: #343541; }
        .message.bot { background-color: #444654; }

        .avatar { width: 36px; height: 36px; border-radius: 4px; margin-right: 15px; display: flex; align-items: center; justify-content: center; font-weight: bold; }
        .user .avatar { background-color: #5436DA; color: white; }
        .bot .avatar { background-color: #10a37f; color: white; }

        .content { line-height: 1.6; width: 100%; overflow-x: hidden; }

        /* Kod Blokları */
        pre { background: #0d0d0d !important; padding: 15px; border-radius: 6px; overflow-x: auto; margin: 10px 0; border: 1px solid #333; }
        code { font-family: 'Consolas', 'Monaco', monospace; font-size: 0.9em; }
        p { margin-bottom: 10px; }

        /* Input Alanı */
        .input-area { padding: 20px; background-color: #343541; border-top: 1px solid #555; position: fixed; bottom: 0; left: 0; right: 0; }
        .input-container { max-width: 800px; margin: 0 auto; position: relative; }
        textarea { width: 100%; background-color: #40414f; border: 1px solid #303038; color: white; padding: 12px 45px 12px 15px; border-radius: 6px; resize: none; outline: none; box-shadow: 0 0 10px rgba(0,0,0,0.1); height: 50px; }
        textarea:focus { border-color: #10a37f; }

        .send-btn { position: absolute; right: 10px; bottom: 10px; background: transparent; border: none; cursor: pointer; color: #ccc; }
        .send-btn:hover { color: #10a37f; }

        .loading { display: none; color: #10a37f; font-size: 0.9em; margin-left: 55px; margin-bottom: 10px; }
    </style>
</head>
<body>

    <div class="chat-container">
        <div style="text-align: center; padding: 20px; border-bottom: 1px solid #555;">
            <h1 class="text-xl font-bold">Spring Boot AI Asistanı 🍃</h1>
            <p class="text-xs text-gray-400">RAG + Fine-Tuned Llama 3.1</p>
        </div>

        <div class="messages" id="messages">
            <div class="message bot">
                <div class="avatar">AI</div>
                <div class="content">
                    Merhaba! Ben Spring Boot konusunda uzmanlaşmış yapay zeka asistanıyım.
                    <br>RestClient, Security, JPA veya konfigürasyonlarla ilgili sorularını sorabilirsin.
                </div>
            </div>
        </div>

        <div class="loading" id="loading">Asistan yazıyor...</div>
        <div style="height: 100px;"></div> </div>

    <div class="input-area">
        <div class="input-container">
            <textarea id="userInput" placeholder="Spring Boot hakkında bir soru sor..." onkeydown="if(event.keyCode===13 && !event.shiftKey){sendMessage(event);}"></textarea>
            <button class="send-btn" onclick="sendMessage()">
                <svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" height="20" width="20" xmlns="http://www.w3.org/2000/svg"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
            </button>
        </div>
    </div>

    <script>
        // Markdown ayarları
        marked.setOptions({
            highlight: function(code, lang) {
                const language = hljs.getLanguage(lang) ? lang : 'plaintext';
                return hljs.highlight(code, { language }).value;
            },
            langPrefix: 'hljs language-'
        });

        async function sendMessage(e) {
            if(e) e.preventDefault();

            const input = document.getElementById('userInput');
            const message = input.value.trim();
            if (!message) return;

            // Kullanıcı mesajını ekle
            addMessage('user', message);
            input.value = '';

            // Loading göster
            document.getElementById('loading').style.display = 'block';

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: message })
                });

                const data = await response.json();

                // Loading gizle
                document.getElementById('loading').style.display = 'none';

                if (data.error) {
                    addMessage('bot', '**Hata:** ' + data.error);
                } else {
                    addMessage('bot', data.response);
                }

            } catch (error) {
                document.getElementById('loading').style.display = 'none';
                addMessage('bot', '**Bağlantı Hatası:** Sunucuya ulaşılamadı.');
            }
        }

        function addMessage(role, text) {
            const messagesDiv = document.getElementById('messages');
            const msgDiv = document.createElement('div');
            msgDiv.className = `message ${role}`;

            const avatar = role === 'user' ? 'Siz' : 'AI';

            // Markdown'ı HTML'e çevir
            const htmlContent = role === 'bot' ? marked.parse(text) : text.replace(/\\n/g, '<br>');

            msgDiv.innerHTML = `
                <div class="avatar">${avatar}</div>
                <div class="content">${htmlContent}</div>
            `;

            messagesDiv.appendChild(msgDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;

            // Kod bloklarını renklendir (yeni eklenenler için)
            document.querySelectorAll('pre code').forEach((block) => {
                hljs.highlightElement(block);
            });
        }
    </script>
</body>
</html>
"""

app = Flask(__name__)
CORS(app)

# Modeli Çıkarım Moduna Al
FastLanguageModel.for_inference(model)

# --- ROUTE 1: ARAYÜZ (HTML) ---
@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_TEMPLATE)

# --- ROUTE 2: CHAT API (RAG + MODEL) ---
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_question = data.get('question', '')

        if not user_question: return jsonify({"error": "Soru boş olamaz"}), 400

        print(f"📩 Web Arayüzünden Gelen Soru: {user_question}")

        # --- ADIM 1: Çeviri ---
        translate_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a strict translator. Translate the technical question below to English.
        RULES: DO NOT answer. ONLY translate. Preserve terms like 'RestClient'.<|eot_id|><|start_header_id|>user<|end_header_id|>
        {user_question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        inputs_trans = tokenizer([translate_prompt], return_tensors="pt").to("cuda")
        outputs_trans = model.generate(**inputs_trans, max_new_tokens=64)
        english_query = tokenizer.batch_decode(outputs_trans)[0].split("assistant<|end_header_id|>")[-1].replace("<|eot_id|>", "").strip()

        print(f"🔍 Aranan (EN): {english_query}")

        # --- ADIM 2: RAG Arama ---
        context_text = ""
        if vector_db:
            docs = vector_db.similarity_search(english_query, k=3)
            context_text = "\n\n".join([d.page_content for d in docs])

        # --- ADIM 3: Cevap Üretimi ---
        rag_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

        Sen bir Spring Boot uzmanısın. Aşağıdaki [BAĞLAM] bilgisini kullanarak cevap ver.

        KURALLAR:
        1. Spring Boot 3.2 'RestClient' arayüzünü kullan.
        2. KESİNLİKLE eski 'RestTemplate' sınıfını kullanma.
        3. Kodların modern 'Fluent API' yapısında olsun.
        4. Cevap dili Türkçe olsun.
        <|eot_id|><|start_header_id|>user<|end_header_id|>

        [BAĞLAM]:
        {context_text}

        SORU:
        {user_question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        inputs = tokenizer([rag_prompt], return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=1024, use_cache=True, temperature=0.3)

        response = tokenizer.batch_decode(outputs)[0]
        clean_response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].replace("<|eot_id|>", "").strip()

        return jsonify({
            "response": clean_response,
        })

    except Exception as e:
        print(f"HATA: {e}")
        return jsonify({"error": str(e)}), 500

# Ngrok Tüneli
ngrok.kill()
try:
    public_url = ngrok.connect(5000, domain=NGROK_STATIC_DOMAIN).public_url
    print(f"\n🚀 SİTE YAYINDA! (Sabit Domain)")
except:
    print(f"\n⚠️ Sabit Domain hatası, rastgele domain deneniyor...")
    public_url = ngrok.connect(5000).public_url

print(f"👉 Arayüze Gitmek İçin Tıkla: {public_url}")

app.run(port=5000)
