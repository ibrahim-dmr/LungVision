    # Akciğer X-Ray Görüntü Sınıflandırma Projesi

    Bu proje, derin öğrenme teknikleri kullanarak akciğer X-ray görüntülerini dokuz farklı kategoriye sınıflandırmayı amaçlamaktadır. Proje, tıbbi görüntü analizi alanında yapay zeka destekli bir teşhis yardımcısı olarak tasarlanmıştır.

    ## Projenin Amacı

    Bu proje, sağlık profesyonellerine akciğer X-ray görüntülerini hızlı ve doğru bir şekilde sınıflandırmada yardımcı olmak için geliştirilmiştir. Özellikle COVID-19 salgını sırasında, hızlı ve güvenilir teşhis araçlarına olan ihtiyacı karşılamak için tasarlanmıştır. Proje, modern derin öğrenme tekniklerini kullanarak yüksek doğruluk oranıyla sınıflandırma yapabilmektedir.

    ## Proje Yapısı

    ```
    lung_X-Ray_Net/
    ├── data/
    │   ├── raw/                  # Ham veri seti
    │   └── processed/            # İşlenmiş veri seti
    ├── src/                      # Kaynak kodlar
    ├── models/                   # Eğitilmiş modeller
    ├── requirements.txt          # Proje bağımlılıkları
    └── README.md                 # Proje dokümantasyonu
    ```

## Kurulum

1. Projeyi klonlayın:
```bash
git clone https://github.com/yourusername/lung_X-Ray_Net.git
cd lung_X-Ray_Net
```

2. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

3. Veri setini indirin ve ön işleme yapın:
```bash
python download_data.py
python preprocess_data.py
```

## Veri Seti

Veri seti, dokuz farklı kategoride sınıflandırılmış göğüs X-ray görüntülerini içermektedir:
- Normal (00_Normal)
- Zatürre (01_Pneumonia)
- Yüksek Yoğunluk (02_High_Density)
- Düşük Yoğunluk (03_Low_Density)
- Tıkayıcı Hastalıklar (04_Obstructive_Diseases)
- Enfeksiyon Hastalıkları (05_Infectious_Diseases)
- Kapsüllü Lezyonlar (06_Encapsulated_Lesions)
- Mediastinal Değişiklikler (07_Mediastinal_Changes)
- Göğüs Değişiklikleri (08_Chest_Changes)

Her bir sınıf, farklı akciğer durumlarını temsil etmektedir ve model, bu durumları yüksek doğruluk oranıyla tespit edebilmektedir.

## Model Mimarisi

Proje, görüntü sınıflandırması için önceden eğitilmiş Swin Transformer modelini (Swin-T) kullanmaktadır. Bu model, modern görüntü işleme ve sınıflandırma görevlerinde yüksek performans göstermektedir.

## Eğitim

Modeli eğitmek için:
```bash
python train.py
```

## Değerlendirme

Modeli değerlendirmek için:
```bash
python evaluate.py
```

## Özellikler

- Yüksek doğruluk oranıyla sınıflandırma
- Hızlı işlem süresi
- Kullanıcı dostu arayüz
- Detaylı raporlama ve analiz
- Sürekli güncellenen model yapısı

## Katkıda Bulunma

Projeye katkıda bulunmak isterseniz, lütfen bir pull request açın. Büyük değişiklikler için, lütfen önce bir issue açarak ne değiştirmek istediğinizi tartışın.

## Lisans

Bu proje MIT Lisansı altında lisanslanmıştır - detaylar için LICENSE dosyasına bakın.

## İletişim

Proje ile ilgili sorularınız veya önerileriniz için lütfen bir issue açın veya e-posta yoluyla iletişime geçin. 