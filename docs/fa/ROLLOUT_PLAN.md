# برنامه فازبندی مهاجرت معماری (بدون اختلال در نسخه فعلی)

## ۱) اهداف و قیود پروژه
- هدف کارایی: `p95 <= 10s`
- معیار کیفیت: Human Judgment با حد پذیرش `>= 3 از 5`
- تعداد محصول در خروجی نهایی: `5`
- زمان مجاز برای fallback LLM: `5s`
- بودجه فراخوانی LLM: متوسط رو به پایین
- استراتژی انتشار: `feature-flag`
- داده رفتاری (click/reorder) فعلاً در دسترس نیست

## ۲) معماری مقصد (Target)
### ۲.۱ مسیر Direct (مسیر سریع)
`Backend -> Router -> Interpret -> Search -> Relevance Guard -> Response Composer`

### ۲.۲ مسیر Abstract/Unclear/Follow-up
`Backend -> Router -> Dialogue Agent -> Clarification -> (در صورت تعیین محصول) -> Direct Path`

### ۲.۳ اصل کلیدی
- `Final LLM` حذف نمی‌شود؛ فقط `شرطی` اجرا می‌شود.
- مسیر Direct در حالت عادی بدون LLM نهایی بسته می‌شود.

## ۳) Feature Flags
- `FF_ROUTER_ENABLED`
- `FF_DIRECT_FASTPATH`
- `FF_CONDITIONAL_FINAL_LLM`
- `DIRECT_FASTPATH_ROLLOUT_PERCENT` (0..100)
- `FINAL_LLM_ROLLOUT_PERCENT` (0..100)
- `FF_INTERPRET_WARMUP`
- `FF_CATEGORY_EMBED_CACHE`
- `FF_CATEGORY_FILTER_GUARD`
- `FF_INTENT_NORMALIZATION`

هر فاز باید با flag جدا فعال شود تا rollback فوری ممکن باشد.

## ۴) تعریف آستانه‌ها (برای فاز ریرنک شرطی)
- `T1`: حداقل امتیاز top1 برای پذیرش بدون LLM نهایی
- `T2`: حداقل فاصله امتیاز top1 و top2 برای اطمینان

مقادیر اولیه پیشنهادی:
- `T1 = 0.55`
- `T2 = 0.08`

این مقادیر بعد از ارزیابی انسانی تنظیم می‌شوند.

## ۵) فازها
## فاز ۰: Baseline و ابزار سنجش
- خروجی موردنیاز:
  - جمع‌آوری `LATENCY_SUMMARY` برای همه سرویس‌ها
  - گزارش تجمعی latency (`scripts/analyze_latency_logs.py`)
- معیار خروج:
  - گزارش baseline هفتگی برای `p50/p95/max`
  - تفکیک زمان Agent, Interpret, Search

## فاز ۱: Quick Wins کم‌ریسک
- تغییرات:
  - warmup اتصال MCP در startup (کاهش cold-start)
  - یکسان‌سازی intentها (`find_best` و معادل‌ها)
  - اصلاح شرط cacheپذیری پاسخ direct
  - کش نتیجه category embedding
- معیار خروج:
  - کاهش محسوس اولین درخواست هر سرویس
  - افزایش نرخ cache hit در direct

## فاز ۲: Router و تفکیک مسیرها
- تغییرات:
  - افزودن Router مرکزی برای تصمیم‌گیری مسیر
  - انتقال queryهای abstract به Dialogue Agent
  - جلوگیری از اجرای search برای abstract تا قبل از انتخاب کاربر
- معیار خروج:
  - no-search برای abstract در لاگ‌ها
  - حفظ کیفیت پاسخ‌های مکالمه‌ای

## فاز ۳: Relevance Guard هیبرید
- تغییرات:
  - امتیاز ترکیبی: lexical + semantic + constraint
  - guard وابسته به intent (اکسسوری را کورکورانه حذف نکند)
  - خروجی نهایی direct به `5` آیتم محدود
- معیار خروج:
  - بهبود کیفیت top-5 در Human Judgment
  - کاهش queryهای «نتیجه بی‌ربط»

## فاز ۴: LLM نهایی شرطی
- تغییرات:
  - اجرای LLM فقط در حالت confidence پایین
  - timeout سخت `5s`
  - محدودسازی ورودی به top-N کوچک (مثل 8)
- معیار خروج:
  - کاهش p95 نسبت به مسیر LLM-all
  - حفظ/بهبود Human Judgment (>= 3/5)

## فاز ۵: Rollout تدریجی و A/B داخلی
- تغییرات:
  - فعال‌سازی flagها به‌صورت مرحله‌ای (10% -> 30% -> 60% -> 100%)
  - مانیتورینگ خطا، latency، کیفیت
- معیار خروج:
  - `p95 <= 10s`
  - Human Judgment پایدار `>= 3/5`

## ۶) سنجه‌های پایش
- Latency:
  - `p50/p95/p99` در سطح `agent_service.chat`, `agent.chat`, `interpret.pipeline`, `search.pipeline`
- Quality:
  - Human Judgment روی نمونه‌های direct/abstract
  - نرخ reformulation (کاربر بلافاصله query جدید اصلاحی می‌زند)
- Stability:
  - error rate
  - timeout rate

## ۷) ریسک‌ها و کنترل ریسک
- ریسک: افت دقت به خاطر guard ساده
  - کنترل: guard وابسته به intent + fallback LLM
- ریسک: افزایش پیچیدگی مسیرها
  - کنترل: feature-flag مستقل + rollback سریع
- ریسک: نبود taxonomy رسمی main/accessory
  - کنترل: استخراج label از query + ruleهای نرم + بازبینی انسانی

## ۸) خروجی قابل تحویل هر فاز
- PR جداگانه
- لیست تغییرات و rollback plan
- گزارش قبل/بعد latency و کیفیت
- وضعیت feature-flagها

## ۹) وضعیت فعلی اجرا (Feature Branch)
- فاز ۱ (Quick Wins): انجام شده
- فاز ۲ (Router تفکیک مسیر): انجام شده
- فاز ۳ (Relevance Guard هیبرید): انجام شده در مسیر direct fastpath
- فاز ۴ (LLM نهایی شرطی): انجام شده (fallback سبک با timeout=5s روی top-N)
- تنظیمات فعال روی برنچ تست:
  - `FF_ROUTER_ENABLED=true`
  - `FF_ABSTRACT_FASTPATH=true`
  - `FF_DIRECT_FASTPATH=true`
  - `FF_CONDITIONAL_FINAL_LLM=true`
  - `DIRECT_FASTPATH_ROLLOUT_PERCENT=100`
  - `FINAL_LLM_ROLLOUT_PERCENT=100`
  - `ROUTER_GUARD_T1=0.55`
  - `ROUTER_GUARD_T2=0.08`
  - `ROUTER_GUARD_MIN_CONFIDENCE=0.58`
  - `FINAL_LLM_TIMEOUT_SECONDS=5`
  - `FINAL_LLM_TOP_N=8`
- نکته عملیاتی:
  - directهای با confidence بالا بدون LLM نهایی پاسخ می‌گیرند.
  - directهای مبهم/low-confidence به مسیر LLM فعلی fallback می‌شوند.
