import pandas as pd
import numpy as np
import difflib
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns

# Kullanılacak Sınıflandırma Modelleri
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# 1. VERİ YÜKLEME VE TEMİZLİK
anket_yolu = "/Users/berkbey/Desktop/Project S.I.D/Pafume user data.csv"
parfum_db_yolu = "/Users/berkbey/Desktop/Project S.I.D/Perfumes_dataset.csv"

df = pd.read_csv(anket_yolu)
df_parfum_db = pd.read_csv(parfum_db_yolu)

df.columns = df.columns.str.strip()
df = df.drop(columns=['Zaman damgası', 'Onaylıyor musunuz?'], errors='ignore')

df = df.rename(columns={
    'Yaşınız?': 'Age', 'Cinsiyetiniz?': 'Gender', 
    'Renklerin kokusu olsa, sizi en iyi yansıtan renk hangisi olurdu?': 'Colour', 
    'Bir ortama girdiğinde nasıl "tanınmak" istersiniz ?': 'Recognize',
    'Hangi koku ailesi size daha yakın?': 'Scent-family', 
    'Sıklıkla kullandığınız bir parfüm var mı, varsa ismi nedir?': 'Parfume',
    'Nasıl bir cilt tipine sahipsiniz?': 'Skin-type',
    'Hangi tür parfümleri tercih edersiniz?': 'Parfume-type', 
    'İdeal parfümünüzün yoğunluğu nasıl olmalı?': 'Parfume-stongivity',
    'Günün hangi zamanlarında parfüm kullanma ihtimalin daha yüksek?': 'Parfume-time', 
    'Bir parfümde hangisi sizin için daha önemli?”': 'Parfume-selection',
    'Geçmişteki herhangi bir anınızla bağdaştırdığınız bir kokudan kısaca bahseder misiniz?  Bu koku nedir?': 'Memo-scent', 
    'Kokuyla bağdaştırdığınız anınızı kısaca anlatabilir misiniz? \n\n(örnek: Maya kokusu, bana annemin hazırladığı o şirin ekmekleri hatırlatır)': 'Memo',
    'Duygunuzun baskınlık seviyesi nedir?': 'Memo-strongivity', 
    'Bu anıyı düşündüğünüzde hangi duyguyu daha baskın hissediyorsunuz?': 'Memo-emotion'
})

df = df.dropna(subset=['Scent-family']).reset_index(drop=True)

# ================= 2. HEDEF (TARGET) GRUPLANDIRMA (4 ANA GRUP) =================
def group_target_scent(family):
    family = str(family).lower()
    if 'amber' in family: return 'Oriental'
    elif 'floral' in family or 'çiçeksi' in family or 'fruity' in family or 'meyveli' in family: return 'Floral'
    elif 'wood' in family or 'odunsu' in family: return 'Woody'
    elif any(x in family for x in ['citrus', 'narenciye', 'water', 'sucul', 'green', 'yeşil', 'aromatic', 'aromatik']): return 'Fresh'
    else: return 'Other'

df['Target_Scent'] = df['Scent-family'].apply(group_target_scent)
le_target = LabelEncoder()
y = le_target.fit_transform(df['Target_Scent'])
target_names = le_target.classes_

# ================= 3. PARFÜM EŞLEŞTİRME (KULLANICININ MEVCUT PARFÜMÜ) =================
df_parfum_db['full_name'] = df_parfum_db['brand'].astype(str).str.lower().str.strip() + ' ' + df_parfum_db['perfume'].astype(str).str.lower().str.strip()
db_list = df_parfum_db[['full_name', 'category']].to_dict('records')

def clean_and_match_perfume(user_input):
    if pd.isna(user_input): return "Bilinmiyor"
    clean_user = str(user_input).lower()
    clean_user = re.sub(r'[^a-z0-9\s]', ' ', clean_user).strip()
    if any(x in clean_user for x in ['yok', 'degisir', 'kullanmiyorum', 'surekli', 'bilmiyorum', 'hatirlamiyorum']): return "Bilinmiyor"
    
    clean_user = clean_user.replace('savage', 'sauvage').replace('galtier', 'gaultier').replace('exilir', 'elixir')
    user_words = set(clean_user.split())
    if not user_words: return "Bilinmiyor"

    best_cat, max_score = "Bilinmiyor", 0
    for item in db_list:
        db_words = set(item['full_name'].split())
        score = len(user_words.intersection(db_words)) / (len(user_words) + 1e-5)
        seq_ratio = difflib.SequenceMatcher(None, clean_user, item['full_name']).ratio()
        if (score + seq_ratio) > max_score:
            max_score, best_cat = (score + seq_ratio), item['category']
            
    if max_score < 0.70: return "Bilinmiyor"
    return best_cat

df['Actual_Scent_Family'] = df['Parfume'].apply(clean_and_match_perfume)

def group_actual_scent(family):
    family = str(family).lower()
    if family == "bilinmiyor": return 'Bilinmiyor'
    elif 'floral' in family or 'çiçek' in family: return 'Floral'
    elif 'wood' in family or 'odunsu' in family or 'fougere' in family: return 'Woody'
    elif 'amber' in family or 'oriental' in family or 'spicy' in family or 'gourmand' in family: return 'Oriental'
    elif any(x in family for x in ['citrus', 'narenciye', 'water', 'sucul', 'green', 'yeşil', 'aromatic', 'fresh']): return 'Fresh'
    else: return 'Diger'

df['Actual_Main_Family'] = df['Actual_Scent_Family'].apply(group_actual_scent)
df = pd.get_dummies(df, columns=['Actual_Main_Family'], prefix='GercekKoku', dtype=int)

# ================= 4. KATEGORİK DÖNÜŞÜMLER & NLP YERİNE DAVRANIŞSAL SKOR =================
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])

def renk_esle(renk):
    renk = str(renk).lower()
    if any(x in renk for x in ['mavi', 'lacivert', 'laci', 'turkuaz']): return 'Mavi'
    elif any(x in renk for x in ['kırmızı', 'bordo']): return 'Kirmizi'
    elif any(x in renk for x in ['yeşil', 'yesil']): return 'Yesil'
    elif any(x in renk for x in ['beyaz', 'vanilya', 'krem', 'bej']): return 'Beyaz'
    elif any(x in renk for x in ['sarı', 'turuncu']): return 'Sari'
    elif any(x in renk for x in ['siyah', 'gri']): return 'Siyah'
    else: return 'Diger'
df['Colour'] = df['Colour'].apply(renk_esle)

df['Parfume-stongivity'] = df['Parfume-stongivity'].astype(str).str.lower().map({'tene yakın':1, 'orta':2, 'güçlü':3, 'i̇z bırakan':4}).fillna(2)

cat_cols = ['Colour', 'Recognize', 'Skin-type', 'Parfume-type', 'Parfume-time', 'Parfume-selection']
df = pd.get_dummies(df, columns=cat_cols, prefix=[c[:3] for c in cat_cols], dtype=int)
df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in df.columns]

# Yaş Grupları
bins = [0, 24, 40, 100]; labels = ['Gen_Z', 'Millennial', 'Gen_X']
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
df = pd.get_dummies(df, columns=['Age_Group'], prefix='AgeGrp', dtype=int)

# Anı Uzunluğu (Davranışsal NLP alternatifi)
def clean_turkish_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-zçğıöşü\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

df['Combined_Memo'] = df['Memo_scent'].fillna("") + " " + df['Memo'].fillna("")
df['Memo_Length'] = df['Combined_Memo'].apply(lambda x: len(clean_turkish_text(x).split()))

# Duygular
df_emotion = pd.get_dummies(df['Memo_emotion'], prefix='Emo', dtype=int)

# ================= 5. MODEL HAZIRLIK VE ÖLÇEKLENDİRME (SCALING) =================
drop_cols = ['Parfume', 'Scent_family', 'Memo_scent', 'Memo', 'Memo_emotion', 
             'Target_Scent', 'Age', 'Actual_Scent_Family', 'Combined_Memo']

X = df.drop(columns=drop_cols, errors='ignore')
X = pd.concat([X, df_emotion], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Lojistik Regresyon, SVM ve KNN gibi modeller için sayısal verileri aynı skalaya getirmeliyiz
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================= 6. TÜM MODELLERİ YARIŞTIRMA (BENCHMARK) =================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42, eval_metric='mlogloss'),
    "SVM (Destek Vektör)": SVC(probability=True, random_state=42),
    "KNN (En Yakın Komşu)": KNeighborsClassifier(n_neighbors=5)
}

results = []

print("\n" + "="*50)
print("🚀 MODELLER YARIŞIYOR... LÜTFEN BEKLEYİN")
print("="*50)

for name, model in models.items():
    # Model eğitimi (Makineleri ölçeklendirilmiş verilerle eğitiyoruz)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results.append({"Model Adı": name, "Doğruluk (Accuracy)": acc})
    print(f"✔️ {name} eğitildi. Başarı: %{acc*100:.2f}")

# Sonuçları Sıralama
results_df = pd.DataFrame(results).sort_values(by="Doğruluk (Accuracy)", ascending=False)

print("\n🏆 ŞAMPİYONLAR LİGİ - SONUÇ TABLOSU:")
print(results_df.to_string(index=False))

# En iyi modeli seçip onun raporunu verelim
best_model_name = results_df.iloc[0]["Model Adı"]
best_model = models[best_model_name]
best_pred = best_model.predict(X_test_scaled)

print(f"\n🥇 KAZANAN MODEL: {best_model_name}")
print("--- DETAYLI SINIFLANDIRMA RAPORU ---")
print(classification_report(y_test, best_pred, target_names=target_names, zero_division=0))

# ================= 7. GÖRSELLEŞTİRME (SADECE AĞAÇ TABANLILAR İÇİN) =================
# Eğer birinci model özellik önemi (Feature Importance) destekliyorsa onu çizdirelim
if hasattr(best_model, "feature_importances_"):
    plt.figure(figsize=(12, 7))
    feat_imps = pd.Series(best_model.feature_importances_, index=X.columns).nlargest(15)
    feat_imps.plot(kind='barh', color='darkorange')
    plt.title(f'Koku Tercihini Belirleyen En Önemli 15 Özellik ({best_model_name})')
    plt.xlabel('Ağırlık (Önem Derecesi)')
    plt.tight_layout()
    plt.savefig('En_Iyi_Model_Onemi.png', dpi=300)
    print("\nAnaliz tamamlandı. 'En_Iyi_Model_Onemi.png' grafiği oluşturuldu.")
else:
    print(f"\nUyarı: Kazanan model ({best_model_name}) Feature Importance desteklemediği için grafik çizdirilemedi.")
    