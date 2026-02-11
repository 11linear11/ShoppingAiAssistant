"""Quick test of the product parser with a real-looking AI response."""
import sys
sys.path.insert(0, '.')

from backend.services.agent_service import AgentService
import json

svc = AgentService()

# Simulate a real AI response (like the screenshot)
test_response = """شورت‌های زنانه مختلفی برای خرید موجود است. لطفا به لیست زیر نگاهی بیندازید:

📦 **شورت زنانه قرمز مایلدا  سایز 2XL**
🏷️ برند: مایلدا
💰 قیمت: ۱۶۰,۰۰۰ تومان
🔗 [مشاهده محصول](url)

📦 **شورت زنانه صورتی مایلدا  سایز L**
🏷️ برند: مایلدا
💰 قیمت: ۱۶۰,۰۰۰ تومان
🔗 [مشاهده محصول](url)

📦 **شورت زنانه صورتی مایلدا  سایز 3XL**
🏷️ برند: مایلدا
💰 قیمت: ۱۶۰,۰۰۰ تومان
🔗 [مشاهده محصول](url)

📦 **شورت زنانه سفید مایلدا  سایز 2XL**
🏷️ برند: مایلدا
💰 قیمت: ۱۶۰,۰۰۰ تومان
🔗 [مشاهده محصول](url)
"""

products = svc._extract_products(test_response)
clean = svc._clean_response_text(test_response, products)

print(f"Products found: {len(products)}")
for p in products:
    print(f"  - {p['name']} | brand={p['brand']} | price={p['price']}")

print(f"\n=== CLEAN RESPONSE ===")
print(repr(clean))
print(f"\n=== What frontend sees ===")
print(f"response: {clean}")
print(f"products: {len(products)} items")
