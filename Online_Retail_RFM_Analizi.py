#########################
# RFM Analizi İle Müşteri Segmentayonu
#########################


#########################
# İş Problemi: İngiltere merkezli perakende şirketi müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama
# stratejileri belirlemek istemektedir.
# Ortak davranışlar sergileyen müşteri segmentleri özelinde pazarlama çalışmaları yapmanın gelir artışı sağlayacağını
# düşünmektedir.
# Segmentlere ayırmak için RFM analizi kullanılacaktır.
#########################


#########################
# Veri Seti Hikayesi
#########################

# Online Retail II isimli veri seti İngiltere merkezli bir perakende şirketinin 01/12/2009 - 09/12/2011 tarihleri
# arasındaki online satış işlemlerini içeriyor. Şirketin ürün kataloğunda hediyelik eşyalar yer almaktadır ve çoğu
# müşterisinin toptancı olduğu bilgisi mevcuttur.

# InvoiceNo: Fatura Numarası (Eğer bu kod C ile başlıyorsa işlemin iptal edildiğini ifade eder)
# StockCode: Ürün kodu (Her bir ürün için eşsiz)
# Description: Ürün ismi
# Quantity: Ürün adedi (Faturalardaki ürünlerden kaçar tane satıldığı)
# InvoiceDate: Fatura tarihi
# UnitPrice: Fatura fiyatı (Sterlin)
# CustomerID: Eşsiz müşteri numarası
# Country: Ülke ismi


#########################
# Görevler
#########################

#########################
# GÖREV 1: Veriyi  Hazırlama ve Anlama (Data Preparing and Understanding)
#########################

import pandas as pd
import datetime as dt
from pandas import DataFrame

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 1000)

# Adım 1: Online Retail II excelindeki 2010-2011 verisini okuyunuz. Oluşturduğunuz dataframe’in kopyasını oluşturunuz.

df_ = pd.read_excel("data_set/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

# Adım 2: Veri setinin betimsel istatistiklerini inceleyiniz.

df.describe().T

# Adım 3: Veri setinde eksik gözlem var mı? Varsa hangi değişkende kaç tane eksik gözlem vardır?

df.isnull().sum()         # Yapılan inceleme sonucunda Customer ID değişkeninde 135080 adet eksik gözlem saptandı.

# Adım 4: Eksik gözlemleri veri setinden çıkartınız.

df.dropna(inplace=True)   # Eksik gözlemler inplace parametresi ile çıkarmamızın sebebi işlemi dataframe'e kaydetmektir.
                          # İşlemi kontrol ettiğimizde eksik değerleri başarılı bir şekilde çıkarttığımızı görüyoruz.

# Adım 5: Eşsiz ürün sayısı kaçtır?

df["Description"].nunique()     # 3896 adet eşsiz ürün vardır.

# Adım 6: Hangi üründen kaçar tane vardır?

df["Description"].value_counts().head()

# Adım 7: En çok sipariş edilen 5 ürünü çoktan aza doğru sıralayınız.

df.groupby("Description").agg({"Quantity": "sum"}).sort_values("Quantity", ascending=False).head()

# Adım 8: Faturalardaki ‘C’ iptal edilen işlemleri göstermektedir. İptal edilen işlemleri veri setinden çıkartınız.

df = df[~df["Invoice"].str.contains("C", na=False)]
df.describe().T

# Adım 9: Fatura başına elde edilen toplam kazancı ifade eden ‘TotalPrice’ adında bir değişken oluşturunuz.

df["TotalPrice"] = df["Quantity"] * df["Price"]
df.head()


#########################
# GÖREV 2: RFM Metriklerinin Hesaplanması
#########################

# Adım 1: Recency, Frequency ve Monetary tanımlarını yapınız.

# Recency (Yenilik): Müşterinin yakınlık tarihine göre satın alma durumu
# Frequency (Sıklık): Her bir müşterinin satın alma sıklığı
# Monetary (Parasal Değer): Her müşterinin bıraktığı toplam parasal değer

# Adım 2: Müşteri özelinde Recency, Frequency ve Monetary metriklerini groupby, agg ve lambda ile hesaplayınız.

# Not: İlk önce analiz yapacağımız tarihi belirlememiz gerekiyor. Çünkü Recency skorunu bulabilmek için söz konusu verinin
# analiz tarihinden müşterilerin son satın alma tarihlerini çıkartmamız gerekiyor.

df["InvoiceDate"].max()                     # Bu kod ile en son yapılan satın alma işleminin tarihini belirliyoruz.
today_date = dt.datetime(2011, 12, 11)      # Analiz yapacağımız tarihi ise 2011.12.11 olarak seçiyoruz.

rfm = df.groupby('Customer ID').agg({'InvoiceDate': lambda InvoiceDate: (today_date - InvoiceDate.max()).days,
                                     'Invoice': lambda Invoice: Invoice.nunique(),
                                     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

# Adım 4: Oluşturduğunuz metriklerin isimlerini recency, frequency ve monetary olarak değiştiriniz.

rfm.columns = ['Recency', 'Frequency', 'Monetary']

rfm.describe().T

# Not: Değişkenlerimizin betimsel istatistiğine baktığımızda Monetary değişkeninin minimum değerini "0" olarak görüyoruz.
# Bunu düzeltmek için monetary değişkenini "0" dan büyük olacak şekilde filtreliyoruz.

rfm = rfm[rfm["Monetary"] > 0]          # Filtreleme işlemini yaptıktan sonra değerin 3.75 yükseldiğini görüyoruz.


#########################
# GÖREV 3: Görev 3: RFM Skorlarının Oluşturulması ve Tek bir Değişkene Çevrilmesi
#########################

# Adım 1: Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çeviriniz.
# Bu skorları recency_score, frequency_score ve monetary_score olarak kaydediniz.

# Not: Recency değeri frequency ve monetary değerlerine göre ters ilişkilidir. Buna göre recency değeri ne kadar düşük
# olursa recency skoru bir o kadar yüksek olacaktır. Bu nedenle frequency ve monetary skorlarının çeyreklik
# hesaplamasında kullanılan labels değerleri büyükten küçüğe doğru sıralanmıştır.
rfm["recency_score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1])

rfm["frequency_score"] = pd.qcut(rfm['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["monetary_score"] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])

# Adım 3: recency_score ve frequency_score’u tek bir değişken olarak ifade ediniz ve RF_SCORE olarak kaydediniz.

rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) +
                   rfm['frequency_score'].astype(str))


#########################
# Görev 5: RF Skorunun Segment Olarak Tanımlanması
#########################

# Adım 1: Oluşturulan RF ve RFM skorları için segment tanımlamaları yapınız.
# Aşağıdaki seg_map yardımı ile skorları segmentlere çeviriniz.

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)

rfm.head()


########################
# Görev 6: Görev 5: Aksiyon Zamanı!
#########################

# Adım 1: Önemli gördüğünü 3 segmenti seçiniz. Bu üç segmenti hem aksiyon kararları açısından hem de segmentlerin yapısı
# açısından(ortalama RFM değerleri) yorumlayınız.

# Adım 2: "Loyal Customers" sınıfına ait customer ID'leri seçerek excel çıktısını alınız.

new_df = pd.DataFrame()
new_df["loyal_customer_id"] = rfm[rfm["segment"] == "loyal_customers"].index

# Not: Müşteri ID'lerinde bulunan ondalıklı değerleri daha sade ve anlaşılır bir şekilde saklamak için temizliyoruz.
new_df["loyal_customer_id"] = new_df["loyal_customers"].astype(int)

new_df.to_excel("loyal_customers.xlsx")


























