import { useState, useRef, useEffect } from 'react';
import './index.css';

// API base URL - uses env variable in dev, relative path in production (nginx proxies /api)
const API_BASE = import.meta.env.VITE_API_URL || '';

const createSessionId = () => {
  const randomUUID = globalThis?.crypto?.randomUUID;
  if (typeof randomUUID === 'function') {
    return randomUUID.call(globalThis.crypto);
  }
  return `sess_${Date.now()}_${Math.random().toString(36).slice(2, 12)}`;
};

// Icons
const SendIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
  </svg>
);

const BotIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="11" width="18" height="10" rx="2"/>
    <circle cx="12" cy="5" r="2"/>
    <path d="M12 7v4"/>
    <line x1="8" y1="16" x2="8" y2="16"/>
    <line x1="16" y1="16" x2="16" y2="16"/>
  </svg>
);

const DIGIT_TRANSLATION = {
  '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
  '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9',
  '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
  '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',
};

const toAsciiDigits = (value = '') => (
  String(value).replace(/[۰-۹٠-٩]/g, (ch) => DIGIT_TRANSLATION[ch] || ch)
);

const toNumber = (value) => {
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  if (typeof value !== 'string') return null;
  const normalized = toAsciiDigits(value).replace(/,/g, '').trim();
  if (!normalized) return null;
  const parsed = Number(normalized);
  return Number.isFinite(parsed) ? parsed : null;
};

const normalizeProduct = (raw) => {
  if (!raw || typeof raw !== 'object') return null;

  const name = (raw.name || raw.product_name || '').toString().trim();
  const brand = (raw.brand || raw.brand_name || '').toString().trim();
  const price = toNumber(raw.price);
  const discountPrice = toNumber(raw.discount_price);
  const discountPercentage = toNumber(raw.discount_percentage);
  const hasDiscount =
    raw.has_discount === true ||
    raw.has_discount === 'true' ||
    (discountPrice !== null && price !== null && discountPrice < price);
  const productUrl = (raw.product_url || raw.url || '').toString().trim();

  if (!name) return null;

  return {
    name,
    brand,
    price,
    discount_price: discountPrice,
    has_discount: hasDiscount,
    discount_percentage: discountPercentage,
    product_url: productUrl,
  };
};

const sanitizeJsonLike = (input) => {
  let text = toAsciiDigits(input || '').trim();
  text = text.replace(/\uFEFF/g, '');
  text = text.replace(/،/g, ',');
  // Some LLMs emit "json" as a literal first line inside fenced blocks.
  text = text.replace(/^\s*json\s*\n/i, '');
  text = text.replace(/,\s*([}\]])/g, '$1');
  text = text.replace(/\bTrue\b/g, 'true').replace(/\bFalse\b/g, 'false').replace(/\bNone\b/g, 'null');
  text = text.replace(/:\s*'([^']*)'/g, ': "$1"');
  text = text.replace(/([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:/g, '$1"$2":');
  // Repair malformed scalars like: "price": 800000.0"
  text = text.replace(/:\s*(-?\d+(?:\.\d+)?)"/g, ': $1');
  text = text.replace(/:\s*(true|false|null)"/gi, ': $1');
  return text;
};

const parseJsonCandidate = (candidate) => {
  if (!candidate) return null;
  const direct = candidate.trim().replace(/^\s*json\s*\n/i, '');
  try {
    return JSON.parse(direct);
  } catch {
    try {
      return JSON.parse(sanitizeJsonLike(direct));
    } catch {
      return null;
    }
  }
};

const normalizeProducts = (items) => {
  if (!Array.isArray(items)) return [];
  const normalized = items.map(normalizeProduct).filter(Boolean);
  const deduped = [];
  const seen = new Set();
  for (const item of normalized) {
    const key = `${item.name}|${item.price ?? ''}|${item.product_url || ''}`;
    if (seen.has(key)) continue;
    seen.add(key);
    deduped.push(item);
  }
  return deduped;
};

const extractProductsFromFields = (text = '') => {
  const objectCandidates = text.match(/\{[\s\S]*?\}/g) || [text];
  const extracted = [];

  const readField = (block, key) => {
    const pattern = new RegExp(
      `["']?${key}["']?\\s*:\\s*(?:"([^"]*)"|'([^']*)'|([^,\\n}\\r]+))`,
      'i'
    );
    const match = block.match(pattern);
    if (!match) return '';
    return (match[1] || match[2] || match[3] || '').trim();
  };

  for (const block of objectCandidates) {
    const maybe = {
      name: readField(block, 'name'),
      product_name: readField(block, 'product_name'),
      brand: readField(block, 'brand'),
      brand_name: readField(block, 'brand_name'),
      price: readField(block, 'price'),
      discount_price: readField(block, 'discount_price'),
      has_discount: readField(block, 'has_discount'),
      discount_percentage: readField(block, 'discount_percentage'),
      product_url: readField(block, 'product_url') || readField(block, 'url'),
    };

    if (maybe.has_discount) {
      maybe.has_discount = /^true$/i.test(String(maybe.has_discount));
    }

    const normalized = normalizeProduct(maybe);
    if (normalized) extracted.push(normalized);
  }

  return normalizeProducts(extracted);
};

const extractProductsFromText = (content) => {
  if (!content) return [];
  const products = [];

  const codeBlockRegex = /(?:```[\t ]*(?:json)?|json```)[\t ]*\n?([\s\S]*?)```/gi;
  const candidates = [];
  let match;
  while ((match = codeBlockRegex.exec(content)) !== null) {
    if (match[1]) candidates.push(match[1]);
  }

  if (candidates.length === 0) {
    const bracketMatch = content.match(/\[[\s\S]*\]/);
    if (bracketMatch) candidates.push(bracketMatch[0]);
    const objectMatch = content.match(/\{[\s\S]*\}/);
    if (objectMatch) candidates.push(objectMatch[0]);
  }

  for (const candidate of candidates) {
    const parsed = parseJsonCandidate(candidate);
    if (parsed) {
      const payload = parsed?.results || parsed?.products || parsed?.data || parsed;
      const rows = Array.isArray(payload) ? payload : [payload];
      for (const row of rows) {
        const normalized = normalizeProduct(row);
        if (normalized) products.push(normalized);
      }
      continue;
    }

    const objectFragments = candidate.match(/\{[\s\S]*?\}/g) || [];
    for (const fragment of objectFragments) {
      const parsedFragment = parseJsonCandidate(fragment);
      if (!parsedFragment) continue;
      const normalized = normalizeProduct(parsedFragment);
      if (normalized) products.push(normalized);
    }

    if (products.length === 0) {
      const fieldProducts = extractProductsFromFields(candidate);
      for (const p of fieldProducts) products.push(p);
    }
  }

  if (products.length === 0) {
    return extractProductsFromFields(content);
  }

  return normalizeProducts(products);
};

const stripJsonBlocks = (content) => {
  if (!content) return '';
  let stripped = content
    .replace(/```[\t ]*(?:json)?[\s\S]*?```/gi, '')
    .replace(/json```[\s\S]*?```/gi, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim();

  // Remove a trailing raw JSON payload if the model emitted it after normal text.
  stripped = stripped.replace(/\n?\s*(\[[\s\S]*\]|\{[\s\S]*\})\s*$/g, '').trim();

  // Clean occasional dangling "json" token.
  stripped = stripped.replace(/\n?\s*json\s*$/i, '').trim();

  return stripped;
};

const formatToman = (value) => {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return `${new Intl.NumberFormat('fa-IR').format(value)} تومان`;
};

const DETAIL_FIELD_ALIASES = {
  name: ['نام محصول', 'محصول', 'product', 'product name', 'نام'],
  brand: ['برند', 'brand'],
  price: ['قیمت', 'price'],
  discount_price: ['قیمت تخفیف‌دار', 'قیمت با تخفیف', 'discount price'],
  discount_percentage: ['درصد تخفیف', 'discount percentage', 'discount percent'],
  category: ['دسته‌بندی', 'دسته بندی', 'category'],
  product_url: ['لینک محصول', 'لینک خرید', 'لینک', 'product link', 'product url', 'url'],
};

const normalizeDetailKey = (rawKey = '') => {
  const key = rawKey
    .toString()
    .replace(/\*/g, '')
    .replace(/[：]/g, ':')
    .trim()
    .toLowerCase();

  for (const [normalized, aliases] of Object.entries(DETAIL_FIELD_ALIASES)) {
    if (aliases.some((alias) => key.includes(alias.toLowerCase()))) {
      return normalized;
    }
  }
  return key;
};

const extractUrlFromText = (value = '') => {
  const markdownMatch = String(value).match(/\[[^\]]+\]\((https?:\/\/[^\s)]+)\)/i);
  if (markdownMatch && markdownMatch[1]) return markdownMatch[1];
  const match = String(value).match(/https?:\/\/[^\s)]+/i);
  return match ? match[0] : '';
};

const parseProductDetailFromText = (content = '') => {
  if (!content) return null;
  const lines = content.split('\n');
  const details = {};
  let parsedFields = 0;

  for (const line of lines) {
    const trimmed = line.trim().replace(/[：]/g, ':');
    if (!trimmed) continue;

    const mdMatch = trimmed.match(/^(?:[-*]\s*)?\*\*(.+?)\*\*\s*:\s*(.+)$/);
    const plainMatch = trimmed.match(/^(?:[-*]\s*)?([^:]{2,})\s*:\s*(.+)$/);
    const match = mdMatch || plainMatch;
    if (!match) continue;

    const key = normalizeDetailKey(match[1]);
    let value = match[2].replace(/\*\*/g, '').trim();
    if (!value) continue;

    if (key === 'price' || key === 'discount_price' || key === 'discount_percentage') {
      const numeric = toNumber(value.replace(/[٪%]/g, '').trim());
      if (numeric !== null) {
        details[key] = numeric;
        parsedFields += 1;
        continue;
      }
    }

    if (key === 'product_url') {
      const url = extractUrlFromText(value);
      details[key] = url || value;
      parsedFields += 1;
      continue;
    }

    details[key] = value;
    parsedFields += 1;
  }

  if (parsedFields < 3 || !details.name) return null;

  return {
    name: String(details.name || '').trim(),
    brand: String(details.brand || '').trim(),
    price: typeof details.price === 'number' ? details.price : toNumber(details.price),
    discount_price: typeof details.discount_price === 'number' ? details.discount_price : toNumber(details.discount_price),
    discount_percentage:
      typeof details.discount_percentage === 'number'
        ? details.discount_percentage
        : toNumber(details.discount_percentage),
    category: String(details.category || '').trim(),
    product_url: String(details.product_url || '').trim(),
  };
};

const stripProductDetailLines = (content = '') => {
  if (!content) return '';
  return content
    .split('\n')
    .filter((line) => {
      const trimmed = line.trim().replace(/[：]/g, ':');
      if (!trimmed) return true;
      if (/^(?:[-*]\s*)?\*\*(.+?)\*\*\s*:\s*(.+)$/.test(trimmed)) return false;
      if (/^(?:[-*]\s*)?(نام محصول|محصول|برند|قیمت|قیمت تخفیف‌دار|قیمت با تخفیف|درصد تخفیف|دسته‌بندی|دسته بندی|لینک محصول|لینک خرید)\s*:\s*(.+)$/i.test(trimmed)) return false;
      return true;
    })
    .join('\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
};

// Typing Indicator - ChatGPT Style
const TypingIndicator = () => (
  <div 
    className="animate-fade-in"
    style={{
      padding: '16px 0',
      background: 'transparent',
    }}
  >
    <div style={{
      maxWidth: '800px',
      margin: '0 auto',
      padding: '0 24px',
      display: 'flex',
      gap: '16px',
      alignItems: 'flex-start'
    }}>
      <div style={{
        width: '32px',
        height: '32px',
        borderRadius: '6px',
        background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: '#fff'
      }}>
        <BotIcon />
      </div>
      <div style={{ flex: 1 }}>
        <div style={{
          fontSize: '13px',
          fontWeight: '600',
          color: 'var(--text-primary)',
          marginBottom: '6px'
        }}>
          دستیار
        </div>
        <div style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
          {[0, 1, 2].map(i => (
            <span key={i} className="typing-dot" style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              background: 'var(--accent)',
            }} />
          ))}
        </div>
      </div>
    </div>
  </div>
);

// Product Table Component
const ProductTable = ({ products }) => {
  if (!products || products.length === 0) return null;

  if (products.length === 1) {
    const p = products[0];
    return (
      <div style={{
        marginTop: '18px',
        borderRadius: '14px',
        border: '1px solid var(--border)',
        background: 'var(--bg-secondary)',
        padding: '14px',
        direction: 'rtl',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: '10px', alignItems: 'flex-start' }}>
          <div style={{ minWidth: 0 }}>
            <div style={{ fontSize: '15px', fontWeight: 600, color: 'var(--text-primary)' }}>{p.name}</div>
            <div style={{ marginTop: '6px', color: 'var(--text-secondary)', fontSize: '13px' }}>
              برند: {p.brand || 'نامشخص'}
            </div>
          </div>
          {(p.product_url || '').trim() ? (
            <a
              href={p.product_url}
              target="_blank"
              rel="noopener noreferrer"
              style={{
                padding: '6px 10px',
                borderRadius: '8px',
                background: 'var(--accent)',
                color: '#fff',
                textDecoration: 'none',
                fontSize: '12px',
                fontWeight: 600,
                whiteSpace: 'nowrap',
              }}
            >
              مشاهده
            </a>
          ) : null}
        </div>

        <div style={{ marginTop: '10px' }}>
          {p.has_discount && p.discount_price ? (
            <div>
              <div style={{ color: 'var(--accent)', fontWeight: 700, fontSize: '16px' }}>
                {formatToman(p.discount_price)}
              </div>
              <div style={{
                color: 'var(--text-muted)',
                fontSize: '12px',
                textDecoration: 'line-through',
                marginTop: '2px'
              }}>
                {formatToman(p.price)}
              </div>
            </div>
          ) : (
            <div style={{ color: 'var(--text-primary)', fontWeight: 700, fontSize: '16px' }}>
              {formatToman(p.price)}
            </div>
          )}
        </div>
      </div>
    );
  }

  const visibleProducts = products.slice(0, 8);

  return (
    <div style={{
    marginTop: '20px',
    borderRadius: '12px',
    overflow: 'hidden',
    border: '1px solid var(--border)',
    direction: 'rtl',
  }}>
    {/* Table Header */}
    <div style={{
      display: 'grid',
      gridTemplateColumns: '2fr 1fr 1fr 80px',
      background: 'var(--bg-tertiary)',
      padding: '12px 16px',
      fontSize: '13px',
      fontWeight: '600',
      color: 'var(--text-secondary)',
      borderBottom: '1px solid var(--border)',
    }}>
      <span>نام محصول</span>
      <span>برند</span>
      <span>قیمت</span>
      <span style={{ textAlign: 'center' }}>لینک</span>
    </div>

    {/* Table Rows */}
    {visibleProducts.map((product, idx) => (
      <div
        key={idx}
        style={{
          display: 'grid',
          gridTemplateColumns: '2fr 1fr 1fr 80px',
          padding: '14px 16px',
          fontSize: '14px',
          color: 'var(--text-primary)',
          background: idx % 2 === 0 ? 'var(--bg-secondary)' : 'var(--bg-primary)',
          borderBottom: idx < visibleProducts.length - 1 ? '1px solid var(--border)' : 'none',
          alignItems: 'center',
          transition: 'background 0.15s ease',
        }}
        onMouseOver={(e) => e.currentTarget.style.background = 'var(--bg-hover)'}
        onMouseOut={(e) => e.currentTarget.style.background = idx % 2 === 0 ? 'var(--bg-secondary)' : 'var(--bg-primary)'}
      >
        {/* Product Name */}
        <span style={{
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
          paddingLeft: '8px',
        }}>
          {product.name}
        </span>

        {/* Brand */}
        <span style={{ color: 'var(--text-secondary)', fontSize: '13px' }}>
          {product.brand || '—'}
        </span>

        {/* Price */}
        <div>
          {product.has_discount && product.discount_price ? (
            <>
              <div style={{ color: 'var(--accent)', fontWeight: '600' }}>
                {formatToman(product.discount_price)}
              </div>
              <div style={{ color: 'var(--text-muted)', fontSize: '12px', textDecoration: 'line-through' }}>
                {formatToman(product.price)}
              </div>
            </>
          ) : (
            <div style={{ fontWeight: '600' }}>
              {product.price
                ? <>{formatToman(product.price)}</>
                : <span style={{ color: 'var(--text-muted)' }}>—</span>
              }
            </div>
          )}
        </div>

        {/* Link */}
        <div style={{ textAlign: 'center' }}>
          {product.product_url ? (
            <a
              href={product.product_url}
              target="_blank"
              rel="noopener noreferrer"
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                width: '32px',
                height: '32px',
                borderRadius: '8px',
                background: 'var(--accent)',
                color: '#fff',
                textDecoration: 'none',
                transition: 'opacity 0.2s',
              }}
              onMouseOver={(e) => e.currentTarget.style.opacity = '0.8'}
              onMouseOut={(e) => e.currentTarget.style.opacity = '1'}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
                <polyline points="15 3 21 3 21 9"/>
                <line x1="10" y1="14" x2="21" y2="3"/>
              </svg>
            </a>
          ) : '—'}
        </div>
      </div>
    ))}

    {/* Discount Badge Row - if any product has discount */}
    {products.some(p => p.has_discount && p.discount_percentage) && (
      <div style={{
        padding: '10px 16px',
        background: 'var(--bg-tertiary)',
        borderTop: '1px solid var(--border)',
        display: 'flex',
        gap: '8px',
        flexWrap: 'wrap',
      }}>
        {products.filter(p => p.has_discount && p.discount_percentage).slice(0, 4).map((p, i) => (
          <span key={i} style={{
            background: '#dc262620',
            color: '#ef4444',
            padding: '3px 10px',
            borderRadius: '20px',
            fontSize: '12px',
            fontWeight: '600',
          }}>
            {Math.round(p.discount_percentage)}% تخفیف — {p.name?.substring(0, 20)}
          </span>
        ))}
      </div>
    )}
  </div>
  );
};

const ProductDetailCard = ({ detail }) => {
  if (!detail) return null;
  const hasDiscount = detail.discount_price !== null && detail.discount_price !== undefined;

  return (
    <div style={{
      marginTop: '16px',
      borderRadius: '14px',
      border: '1px solid var(--border)',
      background: 'var(--bg-secondary)',
      overflow: 'hidden',
      direction: 'rtl',
    }}>
      <div style={{
        padding: '12px 14px',
        borderBottom: '1px solid var(--border)',
        background: 'var(--bg-tertiary)',
      }}>
        <div style={{ fontWeight: 700, fontSize: '15px', color: 'var(--text-primary)' }}>
          {detail.name}
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '120px 1fr' }}>
        <div style={{ padding: '10px 14px', color: 'var(--text-secondary)', borderBottom: '1px solid var(--border)' }}>برند</div>
        <div style={{ padding: '10px 14px', color: 'var(--text-primary)', borderBottom: '1px solid var(--border)' }}>{detail.brand || '—'}</div>

        <div style={{ padding: '10px 14px', color: 'var(--text-secondary)', borderBottom: '1px solid var(--border)' }}>قیمت</div>
        <div style={{ padding: '10px 14px', color: 'var(--text-primary)', borderBottom: '1px solid var(--border)' }}>
          {hasDiscount ? (
            <div>
              <div style={{ color: 'var(--accent)', fontWeight: 700 }}>{formatToman(detail.discount_price)}</div>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)', textDecoration: 'line-through' }}>{formatToman(detail.price)}</div>
            </div>
          ) : (
            <span>{formatToman(detail.price)}</span>
          )}
        </div>

        <div style={{ padding: '10px 14px', color: 'var(--text-secondary)', borderBottom: detail.product_url ? '1px solid var(--border)' : 'none' }}>تخفیف</div>
        <div style={{ padding: '10px 14px', color: 'var(--text-primary)', borderBottom: detail.product_url ? '1px solid var(--border)' : 'none' }}>
          {detail.discount_percentage !== null && detail.discount_percentage !== undefined
            ? `${Math.round(detail.discount_percentage)}%`
            : '—'}
        </div>

        {detail.product_url ? (
          <>
            <div style={{ padding: '10px 14px', color: 'var(--text-secondary)' }}>لینک</div>
            <div style={{ padding: '10px 14px' }}>
              <a
                href={detail.product_url}
                target="_blank"
                rel="noopener noreferrer"
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '6px',
                  color: '#fff',
                  background: 'var(--accent)',
                  textDecoration: 'none',
                  borderRadius: '8px',
                  padding: '6px 10px',
                  fontSize: '12px',
                  fontWeight: 600,
                }}
              >
                مشاهده محصول
              </a>
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
};

// Message Component - Clean Style
const Message = ({ message }) => {
  const isUser = message.role === 'user';
  const hasStructuredProducts = message.products && message.products.length > 0;
  const parsedDetail = !isUser && !hasStructuredProducts ? parseProductDetailFromText(message.content) : null;
  const renderedContent = parsedDetail ? stripProductDetailLines(message.content) : message.content;
  
  return (
    <div 
      className="animate-fade-in"
      style={{
        padding: '12px 0',
        marginTop: !isUser ? '24px' : '0',
      }}
    >
      <div style={{
        maxWidth: '800px',
        margin: '0 auto',
        padding: '0 24px',
        display: 'flex',
        gap: '12px',
        alignItems: 'flex-start',
        direction: isUser ? 'ltr' : 'rtl',
      }}>
        {/* Avatar - Only for AI */}
        {!isUser && (
          <div style={{
            width: '32px',
            height: '32px',
            borderRadius: '50%',
            background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#fff',
            flexShrink: 0,
          }}>
            <BotIcon />
          </div>
        )}

        {/* Content */}
        <div style={{ maxWidth: isUser ? '80%' : '85%', minWidth: 0 }}>
          <div style={{
            color: isUser ? '#fff' : 'var(--text-primary)',
            lineHeight: '1.7',
            fontSize: '15px',
            whiteSpace: 'pre-wrap',
            textAlign: 'right',
            direction: 'rtl',
            ...(isUser ? {
              background: 'var(--accent)',
              padding: '12px 16px',
              borderRadius: '18px 18px 18px 4px',
            } : { padding: '0' })
          }}>
            {renderedContent}
          </div>

          {/* Products - Table */}
          {hasStructuredProducts && (
            <ProductTable products={message.products} />
          )}

          {!hasStructuredProducts && parsedDetail && (
            <ProductDetailCard detail={parsedDetail} />
          )}
        </div>
      </div>
    </div>
  );
};

// Main App
function App() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionId] = useState(() => createSessionId());
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const sendMessage = async () => {
    if (!inputValue.trim() || loading) return;

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: inputValue.trim(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage.content,
          session_id: sessionId,
        }),
      });

      const data = await response.json();
      const responseText = typeof data.response === 'string' ? data.response : '';
      const apiProducts = normalizeProducts(Array.isArray(data.products) ? data.products : []);
      const extractedProducts = apiProducts.length > 0 ? apiProducts : extractProductsFromText(responseText);
      const cleanedResponse = extractedProducts.length > 0
        ? (stripJsonBlocks(responseText) || (extractedProducts.length === 1 ? 'این محصول را برای شما پیدا کردم:' : 'این محصولات را برای شما پیدا کردم:'))
        : responseText;

      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'assistant',
        content: cleanedResponse,
        products: extractedProducts,
      }]);
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'assistant',
        content: 'متأسفانه خطایی رخ داد. لطفاً دوباره تلاش کنید.',
      }]);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  return (
    <div style={{
      height: '100vh',
      display: 'flex',
      flexDirection: 'column',
      background: 'var(--bg-primary)'
    }}>
      {/* Header */}
      <header style={{
        padding: '16px 24px',
        borderBottom: 'none',
        background: 'var(--bg-secondary)',
        display: 'flex',
        alignItems: 'center',
        gap: '12px'
      }}>
        <div style={{
          width: '40px',
          height: '40px',
          borderRadius: '12px',
          background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#fff'
        }}>
          <BotIcon />
        </div>
        <div>
          <h1 style={{
            fontSize: '16px',
            fontWeight: '600',
            color: 'var(--text-primary)',
            margin: 0
          }}>
            دستیار خرید هوشمند
          </h1>
          <span style={{
            fontSize: '12px',
            color: 'var(--accent)'
          }}>
            آنلاین
          </span>
        </div>
      </header>

      {/* Messages */}
      <main style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        overflow: 'auto',
      }}>
        {messages.length === 0 && (
          <div style={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '16px',
            color: 'var(--text-muted)'
          }}>
            <div style={{
              width: '64px',
              height: '64px',
              borderRadius: '20px',
              background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#fff'
            }}>
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="3" y="11" width="18" height="10" rx="2"/>
                <circle cx="12" cy="5" r="2"/>
                <path d="M12 7v4"/>
              </svg>
            </div>
            <div style={{ textAlign: 'center' }}>
              <p style={{ fontSize: '16px', color: 'var(--text-primary)', marginBottom: '4px' }}>
                سلام! 👋
              </p>
              <p style={{ fontSize: '14px' }}>
                من دستیار خرید هوشمند شما هستم. چطور می‌تونم کمکتون کنم؟
              </p>
            </div>
          </div>
        )}

        {messages.map(msg => (
          <Message key={msg.id} message={msg} />
        ))}

        {loading && <TypingIndicator />}

        <div ref={messagesEndRef} />
      </main>

      {/* Input */}
      <footer style={{
        padding: '16px 24px',
        background: 'var(--bg-primary)'
      }}>
        <div style={{
          maxWidth: '800px',
          margin: '0 auto'
        }}>
          <div style={{
            display: 'flex',
            gap: '12px',
            alignItems: 'center',
            background: 'var(--bg-secondary)',
            borderRadius: '24px',
            padding: '4px 4px 4px 4px',
            transition: 'border-color 0.2s ease'
          }}>
            <input
              ref={inputRef}
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && sendMessage()}
              placeholder="پیام خود را بنویسید..."
              disabled={loading}
              style={{
                flex: 1,
                background: 'transparent',
                border: 'none',
                outline: 'none',
                color: 'var(--text-primary)',
                fontSize: '15px',
                padding: '12px 0'
              }}
            />
            <button
              onClick={sendMessage}
              disabled={!inputValue.trim() || loading}
              style={{
                width: '44px',
                height: '44px',
                borderRadius: '50%',
                border: 'none',
                background: inputValue.trim() ? 'var(--accent)' : 'var(--bg-hover)',
                color: inputValue.trim() ? '#fff' : 'var(--text-muted)',
                cursor: inputValue.trim() ? 'pointer' : 'default',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                transition: 'all 0.2s ease'
              }}
            >
              <SendIcon />
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
