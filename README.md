# Sıfır Gürültülü Ekonomi & Bankacılık Haber Botu

**Türkiye Ekonomisi**, **Türkiye Bankacılık Sektörü** ve **Dünya Ekonomisi** başlıklarındaki gelişmeleri birincil kurumlar ve güvenilir finans kaynaklarından otomatik çeken, tık tuzaklarını ve mükerrer haberleri eleyen, **Google Gemini API (`gemini-3.8-flash`)** ile puanlayıp özetleyen ve **Telegram** üzerinden ileten bulut tabanlı haber botu.

---

## Mimari: 4 Katmanlı "Sıfır Gürültü" Hunisi

1. **Küratörlü Kaynak Katmanı (`src/news_bot/sources.py`)**
   * **Resmi Kurumlar & Merkez Bankaları (Tier 1):** TCMB Basın Duyuruları, BDDK Duyuruları, ABD Merkez Bankası (Fed), Avrupa Merkez Bankası (ECB).
   * **Ekonomi ve Finans Basını (Tier 2):** Banka Dünyası (`bankadunyasi.com`), Bloomberg HT, Ekonomim, Dünya Gazetesi, Anadolu Ajansı Ekonomi, TRT Haber Ekonomi, Türkiye Bankacılık/BDDK özel akışı, Financial Times Global Economy, CNBC Economy, BBC Business.
2. **Kural Tabanlı Ön Eleme (`src/news_bot/prefilter.py`)**
   * *"Çeyrek altın kaç TL oldu?"*, *"Emekli promosyonu ne kadar?"*, *"Belli oldu mu?"*, spor, magazin, burç, şans oyunları ve hisse/altcoin pazarlama içeriklerini yapay zekaya göndermeden sıfır maliyetle eler.
   * URL takip parametrelerini (`utm_*`, `fbclid` vb.) temizler ve eski haberleri dışarıda bırakır.
3. **Otorite Öncelikli Tekilleştirme (`src/news_bot/deduplicator.py` & `storage.py`)**
   * `SQLite` ve `RapidFuzz` ile aynı olayı anlatan haberleri karşılaştırır; resmi kurum duyurularına (`Tier 1`) öncelik verir.
   * Farklı dillerdeki aynı haberleri (ör. Fed'in İngilizce duyurusu ile yerel basındaki Türkçe çevirisi) Gemini değerlendirme katmanında `duplicate_of_id` ile yakalar.
4. **Gemini Baş Editör & Akıllı Teslimat (`analyzer.py` & `notifier.py`)**
   * **Sabah 08:00 TSİ Tek Günlük Bülten (`7–9` puan):** Gün boyu ve gece biriken tüm önemli Türkiye ekonomisi, bankacılık ve dünya ekonomisi gelişmeleri her sabah **08:00 TSİ**'de 3 kategori altında derli toplu tek bülten halinde gönderilir.
   * **Olağanüstü Kritik Karar Alarmı (`10/10` puan):** Saat **11:00 TSİ** (TÜİK verileri sonrası), **15:00 TSİ** (TCMB/ECB kararları sonrası) ve **22:30 TSİ** (Fed/Kredi notu kararları sonrası) yapılan kontrollerde yalnızca `10/10` puanlık olağanüstü resmi kararlar varsa anında `🚨 KRİTİK HABER` kartıyla iletilir; aksi halde kanal sessiz kalır.
   * **1–6 Puan / Kapsam Dışı:** Veritabanına işlenip sessizce elenir; bir sonraki taramada tekrar değerlendirilmez.

---

## Adım Adım Kurulum

### 1. Gerekli Anahtarları Alın
1. **Gemini API Anahtarı:** [Google AI Studio](https://aistudio.google.com/apikey) üzerinden ücretsiz bir `GEMINI_API_KEY` oluşturun.
2. **Telegram Bot Token:** Telegram'da `@BotFather` hesabına `/newbot` yazarak botunuzu oluşturun ve size verdiği `TELEGRAM_BOT_TOKEN` değerini kopyalayın. Oluşturduğunuz bota Telegram'dan bir kez `/start` mesajı gönderin.
3. **Telegram Chat ID:**
   * Kişisel mesaj için Telegram'da `@userinfobot` hesabına mesaj atıp `Id` numaranızı alabilir veya tarayıcıda `https://api.telegram.org/bot<TOKEN>/getUpdates` adresini açarak `"chat":{"id":...}` değerini görebilirsiniz.
   * Botu bir gruba veya kanala eklediyseniz ilgili kanal/grup ID'sini (`-100...`) kullanabilirsiniz.

---

### 2. Bilgisayar kapalıyken 7/24 Çalıştırma (GitHub Actions - Ücretsiz)

Proje `.github/workflows/news_bot.yml` iş akışıyla hazır gelir:
1. GitHub'da yeni bir **(tercihen Private / Özel)** depo (repository) oluşturun ve bu klasördeki dosyaları yükleyin:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Economy and banking news bot"
   git branch -M main
   git remote add origin https://github.com/<KULLANICI_ADINIZ>/<DEPO_ADINIZ>.git
   git push -u origin main
   ```
2. GitHub deponuzda **Settings → Secrets and variables → Actions → New repository secret** menüsüne girip şu 3 gizli değişkeni ekleyin:
   * `GEMINI_API_KEY`
   * `TELEGRAM_BOT_TOKEN`
   * `TELEGRAM_CHAT_ID`
3. **Actions** sekmesine geçin, iş akışını etkinleştirin ve hemen test etmek için **Economy & Banking News Bot → Run workflow** butonuna tıklayıp `both` modunu seçerek çalıştırın.
   * Bundan sonra bot **her 30 dakikada bir** otomatik uyanır, SQLite hafızasını GitHub Actions önbelleğinden (`actions/cache`) yükler, kritik haberleri anında atar ve sabah/akşam bülten saatlerinde özet bülteni gönderir.

---

### 3. Yerel Bilgisayarda Test Etme (İsteğe Bağlı)

```powershell
# 1. .env.example dosyasını .env olarak kopyalayıp anahtarlarınızı girin
Copy-Item .env.example .env

# 2. Hiçbir API anahtarı olmadan kaynak çekme ve clickbait filtresini test edin:
.\.venv\Scripts\python.exe main.py --mode fetch-preview

# 3. Telegram'a göndermeden konsolda Gemini puanlamasını ve bülten çıktısını önizleyin:
.\.venv\Scripts\python.exe main.py --mode both --dry-run

# 4. Canlı gönderim testi yapın:
.\.venv\Scripts\python.exe main.py --mode both
```

### Yeni Kaynak Ekleme
Kod değiştirmeden özel RSS kaynağı eklemek veya mevcut bir kaynağı devre dışı bırakmak için `config/custom_sources.json` dosyası oluşturabilirsiniz:
```json
[
  {
    "id": "ozel_kaynak",
    "name": "Özel Ekonomi RSS",
    "url": "https://ornek.com/ekonomi.rss",
    "source_type": "rss",
    "default_category": "TR_ECONOMY",
    "authority_tier": 2,
    "language": "tr",
    "enabled": true
  }
]
```

