# TODO - AiAssistantV3

آخرین به‌روزرسانی: 2026-02-11

## وضعیت

در حال حاضر **تسک الزامی باز** برای این فاز وجود ندارد.

موارد کلیدی انجام‌شده:
- [x] بازسازی و پاکسازی کدهای بلااستفاده
- [x] جایگزینی تست‌های قدیمی با suite جدید و پایدار
- [x] رفع گیر تست API و اجرای موفق تست‌ها به‌صورت فایل‌به‌فایل
- [x] بازنویسی مستندات کامل دوزبانه (EN/FA) با دیاگرام پایپلاین
- [x] به‌روزرسانی `README.md` و `.env.example`
- [x] افزودن مسیر rollback: `scripts/restore_cleanup_backup.sh`

## نتیجه تست فعلی

این فایل‌ها پاس شده‌اند:
- [x] `tests/test_agent_cache.py`
- [x] `tests/test_agent_service.py`
- [x] `tests/test_mcp_client.py`
- [x] `tests/test_pipeline_logger.py`
- [x] `tests/test_backend_api.py`

اگر بعدا تسک جدید تعریف شود، همین فایل به‌روزرسانی می‌شود.
