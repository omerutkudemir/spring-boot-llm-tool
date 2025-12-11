import os
import json
from llama_parse import LlamaParse
from tqdm import tqdm

# Buraya LlamaCloud'dan aldığınız API anahtarını girin
os.environ["LLAMA_CLOUD_API_KEY"] = "llx-....." 

PDF_PATH = "spring-boot-reference.pdf"
OUTPUT_JSON = "rag_data.json"

def main():
    print(f"--- LlamaParse ile Akıllı Dönüştürme Başlıyor: {PDF_PATH} ---")
    
    # LlamaParse ayarları (Tabloları markdown'a çevirir)
    parser = LlamaParse(
        result_type="markdown", 
        verbose=True,
        language="en",
        num_workers=4
    )
    
    # PDF'i işle
    print("Doküman buluta yükleniyor ve işleniyor (Bu işlem tablolara göre 1-2 dk sürebilir)...")
    documents = parser.load_data(PDF_PATH)
    
    final_chunks = []
    
    # Gelen veriyi chunk formatına sok
    for doc in tqdm(documents, desc="Veri işleniyor"):
        # LlamaParse zaten markdown verdiği için ekstra temizliğe çok az ihtiyaç duyar
        final_chunks.append({
            "content": doc.text,
            "metadata": {
                "source": "Spring Boot Reference",
                "page_label": doc.metadata.get('page_label', 'unknown') # Hangi sayfadan geldiği
            }
        })

    # Kaydet
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_chunks, f, ensure_ascii=False, indent=2)
        
    print(f"\n[BAŞARILI] Mükemmel temizlikte veri kaydedildi: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()