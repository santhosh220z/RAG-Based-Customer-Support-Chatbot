-- ============================================================================
-- PostgreSQL Relational Schema: Luxury Textile & Jewelry Store
-- ============================================================================

-- 1. Customers Table
CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(150) NOT NULL,
    email VARCHAR(150) UNIQUE NOT NULL,
    phone VARCHAR(30),
    vip_tier VARCHAR(50) DEFAULT 'Standard',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Products Table (Textiles & Fine Jewelry)
CREATE TABLE IF NOT EXISTS products (
    sku VARCHAR(50) PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    category VARCHAR(100) NOT NULL, -- 'Jewelry: 22K Gold', 'Jewelry: Diamond', 'Textile: Pure Silk', 'Textile: Pashmina'
    material_purity VARCHAR(100),   -- '22K (916 Hallmarked)', 'VVS1 E-Color Diamond', 'Pure Mulberry Silk & Real Zari', '100% Cashmere'
    weight_grams NUMERIC(8, 2),
    price NUMERIC(12, 2) NOT NULL,
    stock_quantity INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Jewelry Certifications (BIS, GIA, IGI)
CREATE TABLE IF NOT EXISTS jewelry_certifications (
    certificate_id VARCHAR(100) PRIMARY KEY,
    sku VARCHAR(50) REFERENCES products(sku),
    issuing_authority VARCHAR(100) NOT NULL, -- 'BIS Hallmarking', 'GIA', 'IGI Antwerp'
    carat_weight NUMERIC(6, 3),
    clarity_grade VARCHAR(50),               -- 'VVS1', 'VVS2', 'IF (Internally Flawless)'
    cut_grade VARCHAR(50),                   -- 'Triple Excellent', 'Ideal Cut'
    gold_hallmark_id VARCHAR(100),
    verified_date DATE
);

-- 4. Customer Orders
CREATE TABLE IF NOT EXISTS orders (
    order_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50) REFERENCES customers(customer_id),
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(80) NOT NULL, -- 'Under Hallmarking', 'Bespoke Tailoring', 'Dispatched - Armored Logistics', 'Delivered'
    tracking_number VARCHAR(100),
    shipping_address TEXT,
    total_amount NUMERIC(12, 2) NOT NULL
);

-- 5. Order Line Items
CREATE TABLE IF NOT EXISTS order_items (
    id SERIAL PRIMARY KEY,
    order_id VARCHAR(50) REFERENCES orders(order_id),
    sku VARCHAR(50) REFERENCES products(sku),
    quantity INT NOT NULL DEFAULT 1,
    unit_price NUMERIC(12, 2) NOT NULL,
    custom_notes TEXT -- E.g., 'Ring size 7 with custom inner engraving "A&S 2026"', 'Blouse stitching included'
);

-- Indices for rapid lookup
CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_orders_tracking ON orders(tracking_number);
CREATE INDEX IF NOT EXISTS idx_cert_sku ON jewelry_certifications(sku);
