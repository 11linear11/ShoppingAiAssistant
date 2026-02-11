# پایپلاین‌ها (فارسی)

## ۱) پایپلاین کامل گفتگو

```mermaid
sequenceDiagram
  participant U as کاربر
  participant FE as فرانت
  participant BE as /api/chat
  participant AS as AgentService
  participant AG as ShoppingAgent
  participant IC as MCP Client
  participant IN as Interpret
  participant SE as Search
  participant EL as Elasticsearch
  participant RD as Redis

  U->>FE: ارسال پیام
  FE->>BE: درخواست chat
  BE->>AS: chat(message, session)
  AS->>RD: بررسی کش سطح ۲
  alt HIT
    AS-->>BE: پاسخ کش‌شده
  else MISS
    AS->>AG: اجرای agent.chat
    AG->>IC: ابزار interpret_query
    IC->>IN: MCP call
    IN-->>IC: query_type + search_params
    IC-->>AG: نتیجه تفسیر

    alt searchable=true
      AG->>IC: ابزار search_products
      IC->>SE: MCP call
      SE->>RD: بررسی کش سطح ۱
      alt MISS
        SE->>EL: کوئری DSL
        EL-->>SE: نتیجه خام
        SE->>SE: rerank
        SE->>RD: ذخیره کش
      end
      SE-->>IC: نتایج جستجو
      IC-->>AG: خروجی ابزار
    end

    AG->>RD: بررسی/ذخیره کش سطح ۳
    AG-->>AS: متن نهایی
    AS->>AS: استخراج محصول + تمیزسازی متن
    AS->>RD: ذخیره احتمالی سطح ۲
    AS-->>BE: پاسخ ساختاریافته
  end
  BE-->>FE: JSON response
```

## ۲) پایپلاین تفسیر (Interpret)
1. نرمال‌سازی متن فارسی.
2. اجرای LLM برای تشخیص نوع کوئری و استخراج پارامترها.
3. ساخت خروجی استاندارد:
   - `searchable`
   - `query_type`
   - `search_params`
4. اگر جستجوپذیر نبود: پیشنهاد و سوال تکمیلی.

## ۳) پایپلاین جستجو (Search)
1. دریافت پارامترهای ساختاریافته.
2. ساخت کلیدهای کش.
3. بررسی negative cache.
4. بررسی search cache.
5. در صورت miss:
   - تولید DSL
   - اجرای جستجو در Elasticsearch
   - rerank
   - ذخیره کش
6. برگرداندن خروجی استاندارد.

## ۴) پایپلاین امبدینگ (Embedding)
1. دریافت متن یا لیست متن.
2. پیش‌پردازش و prefix مدل E5.
3. تولید embedding با sentence-transformers.
4. استفاده اختیاری از کش داخلی در حافظه.
5. بازگشت بردار + متادیتا.

## ۵) پایپلاین لاگینگ
### حالت Debug (`DEBUG_LOG=true`)
- همه استیج‌ها با جزئیات ثبت می‌شوند
- زمان‌بندی start/end ثبت می‌شود

### حالت Non-Debug (`DEBUG_LOG=false`)
- فقط `USER_REQUEST`
- همه خطاها

```mermaid
flowchart LR
  A[log_pipeline] --> B{DEBUG_LOG?}
  B -->|true| C[ثبت کامل]
  B -->|false| D{Error یا USER_REQUEST؟}
  D -->|بله| E[ثبت مینیمال]
  D -->|خیر| F[عدم ثبت]
```

## ۶) پایپلاین رندر فرانت
1. دریافت `response` و `products`.
2. نرمال‌سازی محصولات.
3. اگر `products` خالی بود، استخراج JSON از متن پاسخ.
4. حذف بلاک JSON از متن نمایشی.
5. نمایش:
   - کارت تک‌محصول
   - یا جدول چندمحصول

## ۷) کنترل کارایی و حافظه
- فایل‌های pipeline با rotation
- rotation لاگ docker در compose
- در production: `DEBUG_LOG=false`
