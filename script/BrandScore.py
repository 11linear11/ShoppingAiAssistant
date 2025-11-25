# Ensure 'BrandName' column is entirely strings, handling potential NaN values
cw = {
  "لوازم الکترونیکی": 0.24,
  "لوازم برقی و دیجیتال": 0.17,
  "مد و پوشاک": 0.081,
  "طلا": 0.06,
  "خانه و سبک زندگی": 0.05,
  "خانه و آشپزخانه": 0.04,
  "آرایشی و بهداشتی": 0.04,
  "نوشیدنی": 0.04,
  "لبنیات": 0.03,
  "کالاهای اساسی": 0.03,
  "کودک و نوزاد": 0.03,
  "خواربار و نان": 0.02,
  "بهداشت و سلامت": 0.02,
  "شوینده و مواد ضد عفونی کننده": 0.02,
  "مواد پروتئینی": 0.02,
  "میوه و سبزیجات تازه": 0.02,
  "دستمال و شوینده": 0.015,
  "محصولات سلولزی": 0.01,
  "چاشنی و افزودنی": 0.01,
  "تنقلات": 0.01,
  "کنسرو و غذای آماده": 0.01,
  "صبحانه": 0.01,
  "کنسرو و غذاهای آماده": 0.005,
  "آجیل و خشکبار": 0.005,
  "خشکبار، دسر و شیرینی": 0.005,
  "لوازم تحریر و اداری": 0.005,
  "دسر و شیرینی پزی": 0.002,
  "نان و شیرینی": 0.002
}

import json
import numpy as np

filename = 'export-product-list.csv'

df = pd.read_csv(filename)

brands = df["BrandName"].fillna("").astype(str)
unique_brands, counts = np.unique(brands, return_counts=True)

brand_scores = {}

for brand, count in zip(unique_brands, counts):

    bcat = df[df["BrandName"] == brand]["CategoryName"].fillna("").astype(str)
    bcat_unique, _ = np.unique(bcat, return_counts=True)

    cwt = 0
    for cat in bcat_unique:
        cwt += cw.get(cat, 0)

    if count < 2:
        std_price = 0
    else:
        std_price = df[df["BrandName"] == brand]["Price"].std()
        if np.isnan(std_price):
            std_price = 0

    BrandScore = 0.40 * cwt + 0.25 * count + 0.15 * (1 / (1 + std_price))

    brand_scores[brand] = BrandScore

# ذخیره در فایل JSON
with open("brand_scores.json", "w", encoding="utf-8") as f:
    json.dump(brand_scores, f, ensure_ascii=False, indent=2)

print("فایل brand_scores.json ساخته شد.")

