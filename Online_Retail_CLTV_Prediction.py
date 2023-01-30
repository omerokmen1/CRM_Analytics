##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################

# İngiltere merkezli perakende şirketi satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir. Şirketin
# orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin
# tahmin edilmesi gerekmektedir.

###############################################################
# Veri Seti Hikayesi
###############################################################

# Online Retail II isimli veri seti İngiltere merkezli bir perakende şirketinin 01/12/2009 - 09/12/2011 tarihleri
# arasındaki online satış işlemlerini içeriyor. Şirketin ürün kataloğunda hediyelik eşyalar yer almaktadır ve çoğu
# müşterisinin toptancı olduğu bilgisi mevcuttur.

# InvoiceNo Fatura Numarası (Eğer bu kod C ile başlıyorsa işlemin iptal edildiğini ifade eder)
# StockCode Ürün kodu (Her bir ürün için eşsiz)
# Description Ürün ismi
# Quantity Ürün adedi (Faturalardaki ürünlerden kaçar tane satıldığı)
# InvoiceDate Fatura tarihi
# UnitPrice Fatura fiyatı (Sterlin)
# CustomerID Eşsiz müşteri numarası
# Country Ülke ismi


###############################################################
# GÖREVLER
###############################################################

###############################################################
# GÖREV 1: Veriyi Hazırlama
###############################################################

import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
pd.options.mode.chained_assignment = None

# 1. online_retail.xlsx verisini okuyunuz. Dataframe’in kopyasını oluşturunuz.
df_ = pd.read_excel("data_set/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

# 2. Veri setinde bulunan betimsel istatistikleri inceleyiniz.
df.describe().T

# 3: Faturalardaki ‘C’ iptal edilen işlemleri göstermektedir. İptal edilen işlemleri veri setinden çıkartınız.
df = df[~df["Invoice"].str.contains("C", na=False)]

# 4. Adım 3: Veri setinde eksik gözlem var mı? Varsa hangi değişkende kaç tane eksik gözlem vardır?
df.isnull().sum()       # Yapılan inceleme sonucunda Customer ID (135080) ve Description (1454) değişkenlerinde eksik gözlem saptandı.

# 5: Eksik gözlemleri veri setinden çıkartınız.
df.dropna(inplace=True)     # Dropna fonksiyonu ile eksik gözlemler veri setinden temizlenmiştir.

# 6: Fatura başına elde edilen toplam kazancı ifade eden ‘TotalPrice’ adında bir değişken oluşturunuz.
df["TotalPrice"] = df["Quantity"] * df["Price"]

# 7. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir. Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.
def outlier_thresholds(dataframe, variable):    # Bu fonksiyonun görevi kendisine girilen değişken için eşik değer belirlemektir.
    quartile = dataframe[variable].quantile(0.01)
    quartile2 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile2 - quartile
    up_limit = quartile2 + 1.5 * interquantile_range
    low_limit = quartile - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):   # Bu fonksiyon aykırı değerleri eşik değerlere sabitler.
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit, 0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit, 0)

# 8. "Quantity" ve "Price" değişkenlerinin aykırı değerleri varsa baskılayınız.
replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")


###############################################################
# GÖREV 2: CLTV Veri Yapısının Oluşturulması
###############################################################

# recency: Son satın alma üzerinden geçen zaman. Haftalık. (kullanıcı özelinde)
# T: Müşterinin yaşı. Haftalık. (analiz tarihinden ne kadar süre önce ilk satın alma yapılmış)
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)
# monetary: satın alma başına ortalama kazanç

# 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.
df["InvoiceDate"].max()
analysis_date = dt.datetime(2011, 12, 11)

# 2. recency, T, frequency, ve monetary değerlerinin yer aldığı yeni bir cltv dataframe'i uluşturunuz.
# Bunun için Customer ID'e göre groupby işlemi yaparak istediğimiz değişkenleri hesaplayınız.
cltv_df = df.groupby('Customer ID').agg({
    'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
                     lambda InvoiceDate: (analysis_date - InvoiceDate.min()).days],
     'Invoice': lambda Invoice: Invoice.nunique(),
     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

# 3. Hiyerarşik index problemini çözünüz.
cltv_df.columns = cltv_df.columns.droplevel(0)

# 4. Oluşturduğunuz sütunları 'recency', 'T', 'frequency', 'monetary' olarak isimlendirin.
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']

# 5. İşlem başına ortalama kazanç değerinlerini hesaplayınız.
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

# 6. Frequency değerini 1'den büyük olacak şekilde hesaplayınız.
cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

# 7. recency ve T (Müşterinin yaşı) değerlerini haftalık cinse çeviriniz.
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7


##############################################################
# Görev 3: BG-NBD ve Gamma-Gamma Modellerini Kurarak 6 Aylık CLTV Tahmini Yapılması
##############################################################

# 1. BG/NBD modelini kurunuz.
bgf = BetaGeoFitter(penalizer_coef=0.001)   # BetaGeoFitter fonksiyonu ile bir model nesnesi oluşturuyoruz. Bu model
bgf.fit(cltv_df['frequency'],               # nesnesi ile frequency, recency ve T değerlerini fit metodunu
        cltv_df['recency'],                 # kullanarak veri setine uygun hale getiriyoruz.
        cltv_df['T'])

# 2. GAMMA-GAMMA Modelinin Kurulması
ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

# 3. BG-NBD ve GG modeli ile 6 aylık CLTV'nin hesaplanması
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,      # 6 aylık
                                   freq="W",    # T'nin frekans bilgisi.
                                   discount_rate=0.01)
cltv_df["cltv"] = cltv


###############################################################
# Görev 4: Farklı Zaman Periyotlarından Oluşan CLTV Analizi
###############################################################

# 1: 2010-2011 müşterileri için 1 aylık CLTV hesaplayınız.
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=1,      # 1 aylık
                                   freq="W",    # T'nin frekans bilgisi.
                                   discount_rate=0.01)
cltv_df["cltv"] = cltv
cltv_df.sort_values("cltv", ascending=False)[:10]


# 2: 2010-2011 müşterileri için 12 aylık CLTV hesaplayınız.
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=12,     # 12 aylık
                                   freq="W",    # T'nin frekans bilgisi.
                                   discount_rate=0.01)
cltv_df["cltv"] = cltv
cltv_df.sort_values("cltv", ascending=False)[:10]


###############################################################
# GÖREV 5: CLTV'ye Göre Segmentlerin Oluşturulması
###############################################################

# 1: 2010-2011 müşterileri için 6 aylık CLTV'ye göre tüm müşterilerinizi 4 segmente ayırınız ve grup isimlerini veri setine ekleyiniz.
cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])




















