import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from models.product import Product, db
from utils.tryon_utils import EnhancedVirtualTryOn
from utils.body_detection import BodyDetector
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime
import jwt
import secrets
import time
import sys
from utils.model_processor import ClothingModelProcessor
import json
from utils.pose_processor import PoseProcessor
from utils.clothing_adjuster import ClothingAdjuster
from utils.surface_visualizer import create_visualization
from routes.webrtc_routes import webrtc
import base64

# Windows console encoding fix
if sys.platform.startswith('win'):
    import locale
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)

# Use environment variable for database in production
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///virtualtry.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'obj', 'stl'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize database
db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    price = db.Column(db.Float, nullable=False)
    size_chart = db.Column(db.JSON)
    image_url = db.Column(db.String(200))
    category = db.Column(db.String(50))
    try_on_enabled = db.Column(db.Boolean, default=True)

class Cart(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False)
    quantity = db.Column(db.Integer, default=1)
    size = db.Column(db.String(10))
    price = db.Column(db.Float, nullable=False)
    
    user = db.relationship('User', backref=db.backref('cart_items', lazy=True))
    product = db.relationship('Product')

class Order(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    total_amount = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False)
    shipping_name = db.Column(db.String(100), nullable=False)
    shipping_email = db.Column(db.String(120), nullable=False)
    shipping_phone = db.Column(db.String(20), nullable=False)
    shipping_address = db.Column(db.Text, nullable=False)
    shipping_city = db.Column(db.String(100), nullable=False)
    shipping_state = db.Column(db.String(100), nullable=False)
    shipping_pincode = db.Column(db.String(10), nullable=False)
    payment_method = db.Column(db.String(20), nullable=False)
    payment_status = db.Column(db.String(20), default='pending')
    user = db.relationship('User', backref=db.backref('orders', lazy=True))
    items = db.relationship('OrderItem', backref='order', lazy=True)

class OrderItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(db.Integer, db.ForeignKey('order.id'), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    size = db.Column(db.String(10), nullable=False)
    price = db.Column(db.Float, nullable=False)
    
    product = db.relationship('Product')

db.init_app(app)

def init_db():
    """Initial database setup (only run once for production!)"""
    with app.app_context():
        try:
            db.create_all()
            print("Database initialized successfully!")
        except Exception as e:
            print(f"Error initializing database: {str(e)}")
            db.session.rollback()

# Initialize body detector
body_detector = BodyDetector()

# Initialize processors
pose_processor = PoseProcessor()
clothing_adjuster = ClothingAdjuster()

# Register WebRTC routes
app.register_blueprint(webrtc, url_prefix='/webrtc')

def price_display(amount):
    return f'₹{amount:,.2f}'

@app.context_processor
def utility_processor():
    return dict(price_display=price_display)

# ------------------ ROUTES ------------------

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    # (login code unchanged)
    ...

@app.route('/register', methods=['GET', 'POST'])
def register():
    # (register code unchanged)
    ...

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('Successfully logged out!', 'success')
    return redirect(url_for('home'))

@app.route('/products')
def products():
    # (products code unchanged)
    ...

# (All other routes unchanged – try_on, cart, add_to_cart, place_order, etc.)

# ------------------ MAIN ------------------

if __name__ == '__main__':
    # init_db()  # Uncomment ONLY for first-time database setup
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
