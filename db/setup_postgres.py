"""
Database Setup & Seed Script for Luxury Textile & Jewelry Store
Supports PostgreSQL via DATABASE_URL and local SQLite fallback for seamless local execution.
"""

import os
import sys
from datetime import datetime, date
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    Numeric,
    Text,
    DateTime,
    Date,
    ForeignKey,
    select,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

load_dotenv()

Base = declarative_base()


# ---------------------------------------------------------------------------
# SQLAlchemy Models
# ---------------------------------------------------------------------------
class Customer(Base):
    __tablename__ = "customers"
    customer_id = Column(String(50), primary_key=True)
    name = Column(String(150), nullable=False)
    email = Column(String(150), unique=True, nullable=False)
    phone = Column(String(30))
    vip_tier = Column(String(50), default="Standard")
    created_at = Column(DateTime, default=datetime.utcnow)

    orders = relationship("Order", back_populates="customer")


class Product(Base):
    __tablename__ = "products"
    sku = Column(String(50), primary_key=True)
    name = Column(String(200), nullable=False)
    category = Column(String(100), nullable=False)
    material_purity = Column(String(100))
    weight_grams = Column(Numeric(8, 2))
    price = Column(Numeric(12, 2), nullable=False)
    stock_quantity = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    certifications = relationship("JewelryCertification", back_populates="product")
    order_items = relationship("OrderItem", back_populates="product")


class JewelryCertification(Base):
    __tablename__ = "jewelry_certifications"
    certificate_id = Column(String(100), primary_key=True)
    sku = Column(String(50), ForeignKey("products.sku"))
    issuing_authority = Column(String(100), nullable=False)
    carat_weight = Column(Numeric(6, 3))
    clarity_grade = Column(String(50))
    cut_grade = Column(String(50))
    gold_hallmark_id = Column(String(100))
    verified_date = Column(Date)

    product = relationship("Product", back_populates="certifications")


class Order(Base):
    __tablename__ = "orders"
    order_id = Column(String(50), primary_key=True)
    customer_id = Column(String(50), ForeignKey("customers.customer_id"))
    order_date = Column(DateTime, default=datetime.utcnow)
    status = Column(String(80), nullable=False)
    tracking_number = Column(String(100))
    shipping_address = Column(Text)
    total_amount = Column(Numeric(12, 2), nullable=False)

    customer = relationship("Customer", back_populates="orders")
    items = relationship("OrderItem", back_populates="order")


class OrderItem(Base):
    __tablename__ = "order_items"
    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(50), ForeignKey("orders.order_id"))
    sku = Column(String(50), ForeignKey("products.sku"))
    quantity = Column(Integer, default=1, nullable=False)
    unit_price = Column(Numeric(12, 2), nullable=False)
    custom_notes = Column(Text)

    order = relationship("Order", back_populates="items")
    product = relationship("Product", back_populates="order_items")


# ---------------------------------------------------------------------------
# Database Connection Manager
# ---------------------------------------------------------------------------
def get_engine():
    """Obtain database engine with PostgreSQL support and graceful SQLite fallback."""
    db_url = os.getenv("DATABASE_URL")
    if db_url and db_url.startswith("postgresql://"):
        try:
            engine = create_engine(db_url, pool_pre_ping=True)
            with engine.connect() as conn:
                pass
            print(f"[INFO] Connected successfully to PostgreSQL at: {db_url.split('@')[-1]}")
            return engine
        except Exception as e:
            print(f"[WARNING] Could not connect to live PostgreSQL: {e}")
            print("[INFO] Falling back to local SQLite database: 'textile_jewelry_store.db'...")
    
    # SQLite local file fallback
    sqlite_url = "sqlite:///./textile_jewelry_store.db"
    return create_engine(sqlite_url, connect_args={"check_same_thread": False})


def seed_database(engine):
    """Create tables and seed initial sample data for Textile & Jewelry store."""
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Check if already seeded
    if session.query(Customer).first():
        print("[INFO] Database already contains records. Skipping seed.")
        session.close()
        return

    print("[INFO] Seeding test database with Textile & Jewelry sample records...")

    # 1. Seed Customers
    customers = [
        Customer(
            customer_id="CUST-101",
            name="Priya Sharma",
            email="priya.sharma@example.com",
            phone="+1-555-234-5678",
            vip_tier="Royal Diamond",
        ),
        Customer(
            customer_id="CUST-102",
            name="Ananya Iyer",
            email="ananya.iyer@example.com",
            phone="+1-555-876-5432",
            vip_tier="Gold",
        ),
        Customer(
            customer_id="CUST-103",
            name="David Miller",
            email="david.miller@example.com",
            phone="+1-555-432-1098",
            vip_tier="Standard",
        ),
    ]
    session.add_all(customers)

    # 2. Seed Products
    products = [
        Product(
            sku="JWL-GLD-22K-001",
            name="Royal Heritage 22K Gold Temple Necklace",
            category="Jewelry: 22K Gold",
            material_purity="22K (BIS 916 Hallmarked)",
            weight_grams=48.50,
            price=4250.00,
            stock_quantity=4,
        ),
        Product(
            sku="JWL-DIA-RNG-002",
            name="Solitaire Diamond Engagement Ring (1.5 ct)",
            category="Jewelry: Diamond",
            material_purity="18K White Gold / VVS1 E-Color",
            weight_grams=4.20,
            price=6800.00,
            stock_quantity=8,
        ),
        Product(
            sku="TXT-KAN-SLK-101",
            name="Authentic Kanchipuram Pure Mulberry Silk Saree",
            category="Textile: Pure Silk",
            material_purity="100% Pure Mulberry Silk with Gold Zari Border",
            weight_grams=850.00,
            price=750.00,
            stock_quantity=15,
        ),
        Product(
            sku="TXT-KSH-PSH-202",
            name="Hand-Embroidered Kashmiri Pashmina Shawl",
            category="Textile: Pashmina",
            material_purity="100% Changthangi Cashmere Wool",
            weight_grams=220.00,
            price=490.00,
            stock_quantity=10,
        ),
    ]
    session.add_all(products)

    # 3. Seed Jewelry Certifications
    certifications = [
        JewelryCertification(
            certificate_id="GIA-229871034",
            sku="JWL-DIA-RNG-002",
            issuing_authority="GIA (Gemological Institute of America)",
            carat_weight=1.500,
            clarity_grade="VVS1",
            cut_grade="Triple Excellent",
            gold_hallmark_id="AU-750-GIA",
            verified_date=date(2025, 11, 15),
        ),
        JewelryCertification(
            certificate_id="BIS-916-ND-4491",
            sku="JWL-GLD-22K-001",
            issuing_authority="BIS Hallmarking Bureau",
            carat_weight=None,
            clarity_grade=None,
            cut_grade=None,
            gold_hallmark_id="HUID-916-T78901",
            verified_date=date(2026, 1, 10),
        ),
    ]
    session.add_all(certifications)

    # 4. Seed Orders
    orders = [
        Order(
            order_id="ORD-7821",
            customer_id="CUST-101",
            order_date=datetime(2026, 8, 12, 14, 30),
            status="Dispatched - Armored Logistics",
            tracking_number="ARM-SEC-99812",
            shipping_address="742 Evergreen Terrace, Springfield, OR",
            total_amount=4250.00,
        ),
        Order(
            order_id="ORD-7822",
            customer_id="CUST-102",
            order_date=datetime(2026, 8, 15, 10, 15),
            status="Under Hallmarking & Inspection",
            tracking_number="PENDING-HLM",
            shipping_address="120 Royal Palm Way, Palm Beach, FL",
            total_amount=6800.00,
        ),
        Order(
            order_id="ORD-7823",
            customer_id="CUST-103",
            order_date=datetime(2026, 8, 10, 9, 0),
            status="Delivered",
            tracking_number="FEDEX-EXP-44019",
            shipping_address="500 Madison Ave, New York, NY",
            total_amount=1240.00,
        ),
    ]
    session.add_all(orders)

    # 5. Seed Order Items
    order_items = [
        OrderItem(
            order_id="ORD-7821",
            sku="JWL-GLD-22K-001",
            quantity=1,
            unit_price=4250.00,
            custom_notes="Includes luxury velvet jewelry vault box & BIS certificate.",
        ),
        OrderItem(
            order_id="ORD-7822",
            sku="JWL-DIA-RNG-002",
            quantity=1,
            unit_price=6800.00,
            custom_notes="Ring Size 6.5. Engraving inside band: 'Forever & Always'.",
        ),
        OrderItem(
            order_id="ORD-7823",
            sku="TXT-KAN-SLK-101",
            quantity=1,
            unit_price=750.00,
            custom_notes="Includes matching unstitched pure silk blouse piece.",
        ),
        OrderItem(
            order_id="ORD-7823",
            sku="TXT-KSH-PSH-202",
            quantity=1,
            unit_price=490.00,
            custom_notes="Hand-embroidered floral border.",
        ),
    ]
    session.add_all(order_items)

    session.commit()
    session.close()
    print("[SUCCESS] Database seeded successfully with textile & jewelry records.")


# ---------------------------------------------------------------------------
# Query Helpers for Chatbot
# ---------------------------------------------------------------------------
def lookup_order(order_query: str, engine=None) -> Optional[Dict[str, Any]]:
    """Look up order details by Order ID or Tracking Number."""
    if engine is None:
        engine = get_engine()

    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        clean_query = order_query.strip().upper()
        order = (
            session.query(Order)
            .filter((Order.order_id == clean_query) | (Order.tracking_number == clean_query))
            .first()
        )
        if not order:
            return None

        items_summary = []
        for item in order.items:
            items_summary.append(
                f"- {item.product.name} (Qty: {item.quantity}, Price: ${item.unit_price:,.2f}) | Notes: {item.custom_notes or 'Standard'}"
            )

        return {
            "order_id": order.order_id,
            "customer_name": order.customer.name,
            "vip_tier": order.customer.vip_tier,
            "status": order.status,
            "tracking_number": order.tracking_number,
            "total_amount": float(order.total_amount),
            "order_date": order.order_date.strftime("%B %d, %Y"),
            "items": "\n".join(items_summary),
        }
    finally:
        session.close()


def lookup_certificate(cert_id: str, engine=None) -> Optional[Dict[str, Any]]:
    """Look up authentic jewelry hallmark or diamond certificate by Certificate ID."""
    if engine is None:
        engine = get_engine()

    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        clean_cert = cert_id.strip().upper()
        cert = (
            session.query(JewelryCertification)
            .filter(JewelryCertification.certificate_id == clean_cert)
            .first()
        )
        if not cert:
            return None

        return {
            "certificate_id": cert.certificate_id,
            "issuing_authority": cert.issuing_authority,
            "product_name": cert.product.name,
            "carat_weight": float(cert.carat_weight) if cert.carat_weight else None,
            "clarity_grade": cert.clarity_grade,
            "cut_grade": cert.cut_grade,
            "gold_hallmark_id": cert.gold_hallmark_id,
            "verified_date": cert.verified_date.strftime("%B %d, %Y") if cert.verified_date else "Verified",
        }
    finally:
        session.close()


if __name__ == "__main__":
    eng = get_engine()
    seed_database(eng)
