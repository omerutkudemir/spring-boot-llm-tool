import json
import os
import time
from openai import OpenAI
from tqdm import tqdm
# --- YAPILANDIRMA (Sizin İsteğinize Göre Güncellendi) ---
API_KEY = "sk-proj-....."
# BURAYA KENDİ OPENAI API KEY'İNİZİ YAPIŞTIRIN
INPUT_FILE = "rag_data.json"  
OUTPUT_FILE = "finetune_data.jsonl" 

# Model: GPT-4o-mini (Hızlı, Ucuz ve Akıllı)
MODEL_NAME = "gpt-4o-mini" 

# LİMİT YOK: Tüm dosyayı işlemek için -1 yapıyoruz
PROCESS_LIMIT = -1 

client = OpenAI(api_key=API_KEY)

def generate_synthetic_data(chunk_text):
    """
    GPT-4o-mini'ye metni gönderip 'Instruction Tuning' verisi üretmesini ister.
    """
    
    system_prompt = """
    Sen uzman bir Spring Boot eğitmenisin. Görevin, sana verilen teknik dokümantasyon parçasından 
    "Instruction Fine-Tuning" (Talimat İnce Ayarı) için yüksek kaliteli veri seti üretmektir.
    
    Kurallar:
    1. "instruction" (Soru): Bir yazılımcının karşılaşabileceği gerçekçi ve teknik bir soru olsun.
    2. "output" (Cevap): Dokümandaki bilgiye dayanarak, "Best Practice" vurgusu yapan net bir cevap olsun.
    3. Kod Varsa: Cevapta mutlaka Markdown formatında (```java ... ```) kod örneği olsun.
    4. Dil: Türkçe Soru -> Türkçe Cevap. (Terimler İngilizce kalabilir: 'Bean', 'Context' vb.)
    5. Çıktı Formatı: Sadece geçerli bir JSON nesnesi.
    """

    user_prompt = f"""
    Aşağıdaki metni analiz et. Bu metindeki en önemli bilgiyi öğretecek 
    1 adet "Soru-Cevap" çifti oluştur.
    
    DOKÜMAN METNİ:
    {chunk_text[:4000]} 
    
    İSTENEN JSON FORMATI:
    {{
        "instruction": "Soru...",
        "input": "",
        "output": "Cevap..."
    }}
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={ "type": "json_object" }, 
            temperature=0.6 # Biraz daha tutarlı olması için düşürdük
        )
        
        content = response.choices[0].message.content
        return json.loads(content)
        
    except Exception as e:
        print(f" Hata: {e}")
        return None

def main():
    print(f"--- TAM KAPASİTE VERİ ÜRETİMİ BAŞLIYOR ({MODEL_NAME}) ---\n")
    
    # 1. Dosyayı Oku
    if not os.path.exists(INPUT_FILE):
        print(f"HATA: '{INPUT_FILE}' bulunamadı.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)

    # Limit Kontrolü (-1 ise hepsi)
    if PROCESS_LIMIT == -1:
        chunks_to_process = all_chunks
    else:
        chunks_to_process = all_chunks[:PROCESS_LIMIT]
    
    total_count = len(chunks_to_process)
    print(f"Hedef Parça Sayısı: {total_count}")
    
    # 2. İşlem Döngüsü
    successful_generations = 0
    
    # 'a' (append) modu ile açıyoruz, elektrik kesilirse kaldığı yerden devam eder.
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
        
        for i, chunk in tqdm(enumerate(chunks_to_process), total=total_count, unit="parça"):
            
            text_content = chunk.get("content", "")
            
            # Çok kısa metinleri atla (Gürültü veya başlık olabilir)
            if len(text_content) < 150:
                continue
                
            # OpenAI Çağrısı
            qa_pair = generate_synthetic_data(text_content)
            
            if qa_pair and "instruction" in qa_pair and "output" in qa_pair:
                # JSONL satırı olarak yaz
                f_out.write(json.dumps(qa_pair, ensure_ascii=False) + "\n")
                f_out.flush() # Her satırda diske yazmayı garantile
                successful_generations += 1
            
            # Çok hızlı gidip rate-limit yememek için minik bir bekleme
            # gpt-4o-mini çok hızlıdır, bu bekleme güvenli tarafta kalmak içindir.
            time.sleep(0.2) 

    print(f"\n--- İŞLEM BİTTİ ---")
    print(f"Toplam Başarılı Veri: {successful_generations}")
    print(f"Dosya: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()