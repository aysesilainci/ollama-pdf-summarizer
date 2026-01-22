:

📄 OLLAMA PDF Summarizer – AI Tabanlı Özetleme

Kendi eğittiğim model + OLLaMA ile çalışan RAG tabanlı PDF makale özetleyici

Bu proje, PDF ve metin belgelerini analiz ederek hem klasik özetler hem de chat tabanlı etkileşimli özetler üretir.
Docker ile deploy edilebilen bu sistem, AI tabanlı bilgi erişimi ve özetleme konularında uçtan uca çözüm sunar.

🧠 Özellikler

🔹 PDF veya metin dosyalarını otomatik olarak analiz eder

🔹 Kendi eğitilmiş model + OLLaMA kullanarak özet üretir

🔹 RAG (Retrieval-Augmented Generation) tabanlı bilgi tabanı sorgulama

🔹 Klasik özet ve chat UI ile interaktif özetleme

🔹 Docker ile hızlı deploy ve taşınabilirlik

🔹 Büyük dosya ve multi-PDF desteği



🛠️ Kurulum

Depoyu klonlayın:

git clone https://github.com/aysesilainci/ollama-pdf-summarizer.git
cd ollama-pdf-summarizer


Docker ile çalıştırmak için:

docker-compose up --build


OLLaMA model ve bağımlılıkları container içinde hazır şekilde gelir.

Local olarak Python ile çalıştırmak istersen:

python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

🚀 Kullanım

Klasik özet:

python summarize.py --input data/deneme.pdf


Chat UI ile özetleme:

python chat_ui.py


Chat UI tarayıcı üzerinden interaktif özetleme sağlar.

📁 Proje Yapısı
ollama-pdf-summarizer/
│
├─ data/               # PDF ve metin dosyaları
├─ output/             # Özetlenen dosyalar
├─ backend/            # FastAPI / Django API
├─ docker/             # Docker ve compose dosyaları
├─ models/             # Kendi eğittiğin OLLaMA modeli
├─ summarize.py        # Klasik özet scripti
├─ chat_ui.py          # Chat UI scripti
├─ requirements.txt
└─ README.md

🔍 Nasıl Çalışıyor?

PDF içeriği okunur ve metne dönüştürülür

RAG pipeline ile bilgi tabanı sorgulanır

OLLaMA modeli ile doğal dil özet üretimi yapılır

Kullanıcıya klasik özet veya chat UI üzerinden sunulur

✨ Katkıda Bulunmak

Repo’yu fork’layın

Yeni branch açın (git checkout -b feature-xyz)

Değişikliklerinizi commit edin (git commit -m "Add feature")

Push yapın ve pull request oluşturun

📌 Lisans

MIT License © Ayşe Sıla İnci
