# Akciğer X-Ray Görüntülerinin Derin Öğrenme ile Sınıflandırılması
## İbrahim Halil Demir^1
Fırat Üniversitesi Teknoloji Fakültesi Yazılım Mühendisliği Bölümü
210541017@firat.edu.tr

## Öz
Bu çalışma, akciğer X-ray görüntülerinin derin öğrenme yöntemleri kullanılarak sınıflandırılmasını amaçlamaktadır. Archive 7 veri seti kullanılarak, EfficientNetV2 tabanlı bir model geliştirilmiş ve 9 farklı akciğer durumu için sınıflandırma yapılmıştır. Model, modern derin öğrenme teknikleri ve veri artırma yöntemleri kullanılarak optimize edilmiştir. Çalışmada, modelin performansı detaylı metrikler ve görselleştirmeler ile değerlendirilmiştir. Model, test seti üzerinde %99.21 genel doğruluk oranı elde etmiş ve özellikle bazı sınıflarda (Low Density, Encapsulated Lesions, Mediastinal Changes, Chest Changes) mükemmel sınıflandırma performansı göstermiştir.

## 1. Giriş
Akciğer hastalıklarının teşhisi ve sınıflandırılması, tıbbi görüntüleme alanında kritik bir öneme sahiptir. Geleneksel yöntemler, uzman radyologların manuel değerlendirmesine dayanmaktadır. Ancak, bu süreç zaman alıcı ve bazen öznel olabilmektedir. Derin öğrenme tabanlı otomatik sınıflandırma sistemleri, bu süreci hızlandırabilir ve standardize edebilir.

Bu çalışmada, akciğer X-ray görüntülerini 9 farklı kategoride sınıflandıran bir derin öğrenme modeli geliştirilmiştir. Model, EfficientNetV2 mimarisi kullanılarak oluşturulmuş ve modern optimizasyon teknikleri ile eğitilmiştir. Geliştirilen model, yüksek doğruluk oranları ve tutarlı performans göstermiştir.

## 2. Materyal ve Metot

### 2.1. Veri Seti
Archive 7 veri seti kullanılmıştır. Veri seti 9 farklı sınıf içermektedir:
- 00_Normal
- 01_Pneumonia
- 02_High_Density
- 03_Low_Density
- 04_Obstructive_Diseases
- 05_Infectious_Diseases
- 06_Encapsulated_Lesions
- 07_Mediastinal_Changes
- 08_Chest_Changes

Veri seti, eğitim (%70), doğrulama (%15) ve test (%15) olmak üzere üç alt kümeye ayrılmıştır. Veri seti toplam 1015 test görüntüsü içermektedir.

### 2.2. Model Mimarisi
EfficientNetV2 tabanlı bir model kullanılmıştır. Model mimarisi şu bileşenleri içermektedir:
- Veri ön işleme katmanları
- EfficientNetV2 temel model
- Özelleştirilmiş sınıflandırma başlığı
- Dropout ve Batch Normalization katmanları

### 2.3. Eğitim Süreci
Model eğitimi için aşağıdaki parametreler kullanılmıştır:
- Optimizer: Adam
- Learning Rate: 0.001
- Batch Size: 32
- Epochs: 20
- Loss Function: Cross Entropy Loss

## 3. Deneysel Bulgular

### 3.1. Model Performansı
Model, test seti üzerinde yüksek doğruluk oranları göstermiştir. Sınıf bazında performans metrikleri:

| Sınıf | F1-Skor | Kesinlik | Duyarlılık |
|-------|---------|-----------|------------|
| Normal | 0.988 | 0.990 | 0.985 |
| Pneumonia | 0.981 | 0.975 | 0.987 |
| High Density | 0.995 | 0.990 | 1.000 |
| Low Density | 1.000 | 1.000 | 1.000 |
| Obstructive Diseases | 0.990 | 1.000 | 0.979 |
| Infectious Diseases | 0.989 | 0.989 | 0.989 |
| Encapsulated Lesions | 1.000 | 1.000 | 1.000 |
| Mediastinal Changes | 1.000 | 1.000 | 1.000 |
| Chest Changes | 1.000 | 1.000 | 1.000 |

Genel Performans Metrikleri:
- Genel Doğruluk: %99.21
- Ortalama F1-Skoru: 0.994
- Ortalama Kesinlik: 0.994
- Ortalama Duyarlılık: 0.993

### 3.2. Görselleştirmeler
Projede oluşturulan görselleştirmeler:
1. Sınıf Dağılımı Grafikleri
   - Bar Chart
   - Pie Chart
2. Model Performans Grafikleri
   - Training/Validation Accuracy
   - Training/Validation Loss
3. Değerlendirme Grafikleri
   - Confusion Matrix
   - Accuracy per Class
4. Örnek Görseller
   - Sample Images from Each Class
   - Random Sample Test Results

## 4. Tartışma ve Sonuçlar
Geliştirilen model, akciğer X-ray görüntülerini yüksek doğruluk oranlarıyla sınıflandırabilmektedir. Özellikle bazı sınıflarda (Low Density, Encapsulated Lesions, Mediastinal Changes, Chest Changes) mükemmel sınıflandırma performansı gözlemlenmiştir. Normal ve Pneumonia sınıfları arasında küçük karışıklıklar olmasına rağmen, genel performans oldukça yüksektir.

Modelin güçlü yönleri:
- Yüksek genel doğruluk oranı (%99.21)
- Hızlı çıkarım süresi
- Modern mimari kullanımı
- Tutarlı sınıf performansı

İyileştirme alanları:
- Normal ve Pneumonia sınıfları arasındaki karışıklığın azaltılması
- Obstructive Diseases sınıfı için duyarlılığın artırılması
- Model karmaşıklığının optimizasyonu

## 5. Gelecek Çalışmalar
- Veri artırma tekniklerinin genişletilmesi
- Model mimarisinin optimizasyonu
- Transfer öğrenme yaklaşımlarının denenmesi
- Çoklu modalite entegrasyonu
- Normal ve Pneumonia sınıfları için özel veri artırma tekniklerinin geliştirilmesi

## Kaynaklar
[1] Tan, M., & Le, Q. V. (2021). EfficientNetV2: Smaller Models and Faster Training. arXiv preprint arXiv:2104.00298.
[2] Archive 7 Dataset Documentation
[3] PyTorch Documentation
[4] Medical Image Analysis Literature 