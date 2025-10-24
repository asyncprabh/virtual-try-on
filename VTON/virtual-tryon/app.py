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
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///virtualtry.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')

# Configure allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'obj', 'stl'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize database
from flask_sqlalchemy import SQLAlchemy

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
    with app.app_context():
        try:
            # Drop all tables and recreate them
            db.drop_all()
            db.create_all()
            
            # Create products directory if it doesn't exist
            products_dir = os.path.join(app.root_path, 'static', 'products')
            os.makedirs(products_dir, exist_ok=True)
            
            # Create uploads directory if it doesn't exist
            uploads_dir = os.path.join(app.root_path, 'static', 'uploads')
            os.makedirs(uploads_dir, exist_ok=True)
            
            # Check if we already have any products
            if not Product.query.first():
                # Add sample products
                sample_products = [
                    Product(
                        name='Classic White T-Shirt',
                        description='A premium cotton t-shirt perfect for everyday wear',
                        price=29.99,
                        size_chart={'S': [36, 28], 'M': [38, 30], 'L': [40, 32], 'XL': [42, 34]},
                        image_url='products/19620598_44329692_1000.jpg',
                        category='T-Shirts',
                        try_on_enabled=True
                    ),
                    Product(
                        name='Polo Shirt',
                        description='Classic polo shirt with comfortable fit',
                        price=39.99,
                        size_chart={'S': [36, 28], 'M': [38, 30], 'L': [40, 32], 'XL': [42, 34]},
                        image_url='products/2.png',
                        category='T-Shirts',
                        try_on_enabled=True
                    ),
                    Product(
                        name='Denim Jacket',
                        description='Stylish denim jacket with a modern fit',
                        price=79.99,
                        size_chart={'S': [36, 24], 'M': [38, 25], 'L': [40, 26], 'XL': [42, 27]},
                        image_url='products/7.png',
                        category='Jackets',
                        try_on_enabled=True
                    ),
                    Product(
                        name='Formal Black Blazer',
                        description='Elegant black blazer for formal occasions',
                        price=129.99,
                        size_chart={'S': [36, 28], 'M': [38, 29], 'L': [40, 30], 'XL': [42, 31]},
                        image_url='products/R.jpeg',
                        category='Formal Wear',
                        try_on_enabled=True
                    ),
                    Product(
                        name='Casual Hoodie',
                        description='Comfortable hoodie for casual wear',
                        price=49.99,
                        size_chart={'S': [36, 26], 'M': [38, 27], 'L': [40, 28], 'XL': [42, 29]},
                        image_url='products/6.png',
                        category='Casual Wear',
                        try_on_enabled=True
                    )
                ]
                
                for product in sample_products:
                    db.session.add(product)
            
            # Check if we already have any users
            if not User.query.first():
                # Create a test user
                test_user = User(
                    name='Test User',
                    email='test@example.com',
                    password=generate_password_hash('password123')
                )
                db.session.add(test_user)
            
            # Commit all changes
            db.session.commit()
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
    # You can modify this function to use different currency symbols and formats
    currency_symbol = '₹'  # Indian Rupee symbol
    return f'{currency_symbol}{amount:,.2f}'

@app.context_processor
def utility_processor():
    return dict(price_display=price_display)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            email = request.form.get('email')
            password = request.form.get('password')
            
            if not email or not password:
                flash('Please provide both email and password', 'error')
                return render_template('login.html')
            
            user = User.query.filter_by(email=email).first()
            if user and check_password_hash(user.password, password):
                session['user_id'] = user.id
                flash('Successfully logged in!', 'success')
                return redirect(url_for('products'))
            
            flash('Invalid email or password', 'error')
            return render_template('login.html')
        except Exception as e:
            app.logger.error(f"Login error: {str(e)}")
            flash('An error occurred during login', 'error')
            return render_template('login.html')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            name = request.form.get('name')
            email = request.form.get('email')
            password = request.form.get('password')
            
            if not all([name, email, password]):
                flash('Please fill in all fields', 'error')
                return render_template('register.html')
            
            if User.query.filter_by(email=email).first():
                flash('Email already registered', 'error')
                return render_template('register.html')
            
            user = User(
                name=name,
                email=email,
                password=generate_password_hash(password)
            )
            db.session.add(user)
            db.session.commit()
            
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Registration error: {str(e)}")
            flash('An error occurred during registration', 'error')
            return render_template('register.html')
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('Successfully logged out!', 'success')
    return redirect(url_for('home'))

@app.route('/products')
def products():
    if 'user_id' not in session:
        flash('Please login to view products', 'error')
        return redirect(url_for('login'))
    products = Product.query.all()
    return render_template('products.html', products=products)

@app.route('/cart')
def cart():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    cart_items = []
    total_price = 0
    
    # Get cart items from database
    db_cart_items = Cart.query.filter_by(user_id=session['user_id']).all()
    
    for cart_item in db_cart_items:
        product = cart_item.product
        if product:
            # Get size information from the product's size chart
            sizes = list(product.size_chart.keys()) if product.size_chart else ['S', 'M', 'L', 'XL']
            
            cart_items.append({
                'id': product.id,
                'name': product.name,
                'description': product.description,
                'base_price': product.price,
                'unit_price': cart_item.price,
                'unit_price_display': price_display(cart_item.price),
                'price': cart_item.price * cart_item.quantity,
                'price_display': price_display(cart_item.price * cart_item.quantity),
                'image_url': product.image_url,
                'sizes': sizes,
                'selected_size': cart_item.size,
                'quantity': cart_item.quantity
            })
            total_price += cart_item.price * cart_item.quantity
    
    return render_template('cart.html', 
                           cart_items=cart_items,
                           total_price_display=price_display(total_price))

@app.route('/add_to_cart', methods=['POST'])
def add_to_cart():
    if 'user_id' not in session:
        return jsonify({'error': 'Please login first'}), 401
    
    product_id = request.form.get('product_id')
    if not product_id:
        return jsonify({'error': 'Product ID is required'}), 400
        
    # Check if product exists
    product = Product.query.get(product_id)
    if not product:
        return jsonify({'error': 'Product not found'}), 404
    
    # Check if product is already in cart
    existing_item = Cart.query.filter_by(
        user_id=session['user_id'],
        product_id=product_id
    ).first()
    
    if existing_item:
        return jsonify({'success': True, 'message': 'Product is already in cart'})
    
    # Add new item to cart
    # Default to size 'M' and calculate price with size multiplier
    size = 'M'
    size_multiplier = 1.1  # M size multiplier
    price = product.price * size_multiplier
    
    cart_item = Cart(
        user_id=session['user_id'],
        product_id=product_id,
        quantity=1,
        size=size,
        price=price
    )
    
    try:
        db.session.add(cart_item)
        db.session.commit()
        return jsonify({
            'success': True,
            'message': 'Product added to cart',
            'cart_item': {
                'product_name': product.name,
                'quantity': 1,
                'size': size,
                'price': price,
                'price_display': price_display(price)
            }
        })
    except Exception as e:
        db.session.rollback()
        app.logger.error(f'Error adding to cart: {str(e)}')
        return jsonify({'error': 'Error adding product to cart'}), 500

@app.route('/remove_from_cart', methods=['POST'])
def remove_from_cart():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    product_id = request.form.get('product_id')
    if product_id:
        cart_item = Cart.query.filter_by(
            user_id=session['user_id'],
            product_id=product_id
        ).first()
        
        if cart_item:
            try:
                db.session.delete(cart_item)
                db.session.commit()
                flash('Product removed from cart!', 'success')
            except Exception as e:
                db.session.rollback()
                app.logger.error(f'Error removing from cart: {str(e)}')
                flash('Error removing product from cart!', 'error')
    
    return redirect(url_for('cart'))

@app.route('/update_size', methods=['POST'])
def update_size():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    product_id = request.form.get('product_id')
    size = request.form.get('size')
    
    if product_id and size:
        session[f'size_{product_id}'] = size
        session.modified = True
        
        # Calculate new price
        product = Product.query.get(product_id)
        if product:
            size_multipliers = {'S': 1.0, 'M': 1.1, 'L': 1.2, 'XL': 1.3}
            new_price = product.price * size_multipliers.get(size, 1.0)
            return jsonify({
                'success': True,
                'new_price': new_price,
                'price_display': price_display(new_price)
            })
    
    return jsonify({'error': 'Invalid request'}), 400

@app.route('/update_quantity', methods=['POST'])
def update_quantity():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    product_id = request.form.get('product_id')
    quantity = request.form.get('quantity')
    
    if product_id and quantity:
        try:
            quantity = int(quantity)
            if 1 <= quantity <= 10:  # Validate quantity range
                # Get cart item
                cart_item = Cart.query.filter_by(
                    user_id=session['user_id'],
                    product_id=product_id
                ).first()
                
                if cart_item:
                    cart_item.quantity = quantity
                    db.session.commit()
                    
                    # Calculate new price
                    total_price = cart_item.price * quantity
                    
                    return jsonify({
                        'success': True,
                        'unit_price': cart_item.price,
                        'unit_price_display': price_display(cart_item.price),
                        'total_price': total_price,
                        'total_price_display': price_display(total_price)
                    })
                
                return jsonify({'error': 'Item not found in cart'}), 404
            
            return jsonify({'error': 'Invalid quantity'}), 400
        except ValueError:
            return jsonify({'error': 'Invalid quantity format'}), 400
@app.route('/place-order', methods=['POST'])
def place_order():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get cart items
    cart_items = Cart.query.filter_by(user_id=session['user_id']).all()
    if not cart_items:
        flash('Your cart is empty!', 'error')
        return redirect(url_for('cart'))
    
    # Get form data
    name = request.form.get('name')
    email = request.form.get('email')
    phone = request.form.get('phone')
    address = request.form.get('address')
    city = request.form.get('city')
    state = request.form.get('state')
    pincode = request.form.get('pincode')
    payment_method = request.form.get('payment_method')
    
    if not all([name, email, phone, address, city, state, pincode, payment_method]):
        flash('Please fill in all required fields', 'error')
        return redirect(url_for('order_details'))
    
    # Calculate total amount
    total_amount = sum(item.price * item.quantity for item in cart_items)
    
    # Create order
    order = Order(
        user_id=session['user_id'],
        total_amount=total_amount,
        created_at=datetime.now(),
        shipping_name=name,
        shipping_email=email,
        shipping_phone=phone,
        shipping_address=address,
        shipping_city=city,
        shipping_state=state,
        shipping_pincode=pincode,
        payment_method=payment_method,
        payment_status='pending'
    )
    
    try:
        # Add order to database
        db.session.add(order)
        db.session.flush()  # This assigns an ID to the order
        
        # Create order items
        for cart_item in cart_items:
            order_item = OrderItem(
                order_id=order.id,
                product_id=cart_item.product_id,
                quantity=cart_item.quantity,
                size=cart_item.size,
                price=cart_item.price
            )
            db.session.add(order_item)
        
        # Clear cart
        for item in cart_items:
            db.session.delete(item)
        
        db.session.commit()
        
        # Show appropriate message based on payment method
        if payment_method == 'cod':
            flash('Order placed successfully! You will pay ₹{} on delivery.'.format(total_amount), 'success')
        elif payment_method == 'upi':
            flash('Order placed successfully! Please complete UPI payment of ₹{}.'.format(total_amount), 'success')
        else:  # card payment
            flash('Order placed successfully! Please complete card payment of ₹{}.'.format(total_amount), 'success')
        
        return redirect(url_for('my_orders'))
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f'Error placing order: {str(e)}')
        flash('Error placing order. Please try again.', 'error')
        return redirect(url_for('cart'))

@app.route('/product/<int:product_id>')
def product_detail(product_id):
    if 'user_id' not in session:
        flash('Please login to view product details', 'error')
        return redirect(url_for('login'))
    product = Product.query.get_or_404(product_id)
    return render_template('product_detail.html', product=product)

@app.route('/try_on/<int:product_id>', methods=['GET', 'POST'])
def try_on(product_id):
    if 'user_id' not in session:
        flash('Please login to use virtual try-on', 'error')
        return redirect(url_for('login'))
    
    product = Product.query.get_or_404(product_id)
    result_image = None
    error = None
    
    if request.method == 'POST':
        try:
            print("\n=== Starting Virtual Try-On Process ===")
            
            if 'photo' not in request.files:
                raise ValueError('No photo uploaded')
            
            photo = request.files['photo']
            if photo.filename == '':
                raise ValueError('No photo selected')
            
            if photo and allowed_file(photo.filename):
                # Create upload directory if it doesn't exist
                upload_dir = os.path.join(app.root_path, 'static', 'uploads')
                os.makedirs(upload_dir, exist_ok=True)
                
                # Save uploaded photo with timestamp
                timestamp = int(time.time())
                filename = f"{timestamp}_{secure_filename(photo.filename)}"
                photo_path = os.path.join(upload_dir, filename)
                photo.save(photo_path)
                print(f"✓ Saved user photo to: {photo_path}")
                
                # Get product image path
                product_path = os.path.join(app.root_path, 'static', product.image_url)
                if not os.path.exists(product_path):
                    raise ValueError(f"Product image not found: {product.image_url}")
                print(f"✓ Found product image at: {product_path}")
                
                # Generate result path
                result_filename = f'result_{timestamp}_{secure_filename(photo.filename)}'
                result_path = os.path.join(upload_dir, result_filename)
                print(f"✓ Will save result to: {result_path}")
                
                # Process try-on
                tryon_system = EnhancedVirtualTryOn()
                success = tryon_system.try_on(photo_path, product_path, result_path)
                
                if success and os.path.exists(result_path):
                    # Get relative path for URL
                    result_rel_path = os.path.relpath(result_path, os.path.join(app.root_path, 'static'))
                    result_rel_path = result_rel_path.replace('\\', '/')  # Fix Windows paths
                    result_image = url_for('static', filename=result_rel_path)
                    print(f"✓ Success! Result available at: {result_image}")
                    flash('Virtual try-on completed successfully!', 'success')
                else:
                    raise ValueError('Failed to generate try-on result')
            else:
                raise ValueError('Invalid file type. Please use .png, .jpg, .jpeg, or .gif files.')
                
        except Exception as e:
            error = str(e)
            print(f"✗ Error: {error}")
            flash(f'Error: {error}', 'error')
    
    return render_template('try_on.html', 
                         product=product,
                         result_image=result_image,
                         error=error)

@app.route('/try_on_realtime/<int:product_id>', methods=['POST'])
def try_on_realtime(product_id):
    try:
        if 'frame' not in request.files:
            return jsonify({'error': 'No frame provided'}), 400
            
        frame_file = request.files['frame']
        if frame_file.filename == '':
            return jsonify({'error': 'No frame selected'}), 400

        # Get the product
        product = Product.query.get_or_404(product_id)
        
        # Save the frame temporarily
        frame_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_frame.jpg')
        frame_file.save(frame_path)
        
        # Read the frame
        frame = cv2.imread(frame_path)
        if frame is None:
            return jsonify({'error': 'Could not read frame'}), 400
            
        # Get the clothing image path
        clothing_path = os.path.join(app.root_path, 'static', product.image_url)
        
        # Process the frame with body detection and clothing overlay
        try:
            processed_frame = body_detector.process_frame(frame, clothing_path)
            
            # Save the processed frame
            result_filename = f'result_{int(time.time())}.jpg'
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'results', result_filename)
            cv2.imwrite(result_path, processed_frame)
            
            # Return the processed image path
            return jsonify({
                'result_image': url_for('static', filename=f'uploads/results/{result_filename}')
            })
            
        except Exception as e:
            return jsonify({'error': f'Processing error: {str(e)}'}), 500
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/start_virtual_tryon', methods=['POST'])
def start_virtual_tryon():
    """Start real-time virtual try-on session"""
    if 'clothing' not in request.files:
        return jsonify({'error': 'No clothing image provided'}), 400
        
    file = request.files['clothing']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format'}), 400

    # Save clothing image
    filename = secure_filename(file.filename)
    clothing_path = os.path.join(app.config['UPLOAD_FOLDER'], 'clothing', filename)
    os.makedirs(os.path.dirname(clothing_path), exist_ok=True)
    file.save(clothing_path)
    
    # Store clothing path in session
    session['current_clothing'] = clothing_path
    
    # Start pose processor
    pose_processor.start_processing()
    
    return jsonify({'success': True, 'message': 'Virtual try-on session started'})

@app.route('/stop_virtual_tryon', methods=['POST'])
def stop_virtual_tryon():
    """Stop virtual try-on session"""
    pose_processor.stop_processing()
    session.pop('current_clothing', None)
    return jsonify({'success': True, 'message': 'Virtual try-on session stopped'})

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Process a single frame for virtual try-on with 3D pose estimation"""
    try:
        # Get frame data
        frame_data = request.get_json()
        frame = np.array(frame_data['frame'])
        
        # Get current clothing image
        clothing_path = session.get('current_clothing')
        if not clothing_path:
            return jsonify({'error': 'No clothing selected'}), 400
            
        clothing_img = cv2.imread(clothing_path, cv2.IMREAD_UNCHANGED)
        if clothing_img is None:
            return jsonify({'error': 'Failed to load clothing image'}), 500
            
        # Process frame with 3D pose estimation
        processed_frame, fit_metrics, heatmap = clothing_adjuster.adjust_clothing(
            frame,
            clothing_img,
            None  # landmarks will be handled inside adjust_clothing now
        )
        
        if processed_frame is None:
            return jsonify({'error': 'Failed to detect pose'}), 400
        
        # Encode processed frame and heatmap
        _, frame_buffer = cv2.imencode('.jpg', processed_frame)
        frame_data = base64.b64encode(frame_buffer).decode('utf-8')
        
        _, heatmap_buffer = cv2.imencode('.jpg', heatmap)
        heatmap_data = base64.b64encode(heatmap_buffer).decode('utf-8')
        
        # Prepare response with 3D transformation data
        response_data = {
            'processed_frame': frame_data,
            'heatmap': heatmap_data,
            'fit_analysis': {
                'overall_score': fit_metrics.overall_score if fit_metrics else 0,
                'shoulder_alignment': fit_metrics.shoulder_alignment if fit_metrics else 0,
                'chest_fit': fit_metrics.chest_fit if fit_metrics else 0,
                'waist_fit': fit_metrics.waist_fit if fit_metrics else 0,
                'arm_holes': fit_metrics.arm_holes if fit_metrics else 0,
                'problem_areas': fit_metrics.problem_areas if fit_metrics else []
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_result/<path:filename>')
def download_result(filename):
    try:
        # Construct the full path to the result image
        upload_dir = os.path.join(app.root_path, 'static', 'uploads')
        return send_from_directory(
            upload_dir, 
            filename,
            as_attachment=True,
            download_name='virtual-tryon-result.jpg'
        )
    except Exception as e:
        flash('Error downloading the result: ' + str(e), 'error')
        return redirect(url_for('index'))

@app.route('/process_3d_model', methods=['POST'])
def process_3d_model():
    if 'model' not in request.files:
        return jsonify({'error': 'No model file provided'}), 400
    
    file = request.files['model']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format'}), 400

    # Save the uploaded model
    filename = secure_filename(file.filename)
    model_path = os.path.join(app.config['UPLOAD_FOLDER'], 'models', filename)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    file.save(model_path)

    try:
        # Process the 3D model
        processor = ClothingModelProcessor(model_path)
        processor.load_model()
        
        # Handle body mesh if provided
        if 'body_mesh' in request.files:
            body_mesh_file = request.files['body_mesh']
            body_mesh_filename = secure_filename(body_mesh_file.filename)
            body_mesh_path = os.path.join(app.config['UPLOAD_FOLDER'], 'models', 'body_' + body_mesh_filename)
            body_mesh_file.save(body_mesh_path)
            processor.load_body_mesh(body_mesh_path)

        # Handle texture if provided
        if 'texture' in request.files:
            texture_file = request.files['texture']
            texture_filename = secure_filename(texture_file.filename)
            texture_path = os.path.join(app.config['UPLOAD_FOLDER'], 'textures', texture_filename)
            os.makedirs(os.path.dirname(texture_path), exist_ok=True)
            texture_file.save(texture_path)
            processor.apply_texture_mapping(texture_path)

        # Apply processing steps based on request parameters
        if 'scale' in request.form:
            scale_factor = float(request.form['scale'])
            processor.adjust_scale(scale_factor)
        
        if 'smooth' in request.form:
            iterations = int(request.form['smooth'])
            lambda_factor = float(request.form.get('lambda_factor', 0.5))
            processor.smooth_surface(iterations, lambda_factor)

        # Handle mesh deformation if constraint points are provided
        if 'constraint_points' in request.form and 'target_points' in request.form:
            constraint_points = np.array(json.loads(request.form['constraint_points']))
            target_points = np.array(json.loads(request.form['target_points']))
            weight = float(request.form.get('deform_weight', 1.0))
            processor.deform_to_body(constraint_points, target_points, weight)

        # Export processed model
        output_filename = f"processed_{filename}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'models', output_filename)
        processor.export_model(output_path)

        return jsonify({
            'success': True,
            'message': 'Model processed successfully',
            'processed_model': output_filename
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_surface', methods=['POST'])
def generate_surface():
    """Generate a 3D surface visualization"""
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'visualizations')
        os.makedirs(output_dir, exist_ok=True)
        
        # Get visualization type from request
        use_plotly = request.form.get('use_plotly', 'false').lower() == 'true'
        
        # Generate visualization
        output_path = create_visualization(output_dir, use_plotly)
        
        # Get the filename from the path
        filename = os.path.basename(output_path)
        
        return jsonify({
            'success': True,
            'message': 'Surface visualization generated successfully',
            'file_url': url_for('static', filename=f'uploads/visualizations/{filename}')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.after_request
def add_header(response):
    """Ensure proper caching for dynamic images"""
    if 'image' in response.mimetype:
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return response

@app.route('/order-details')
def order_details():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get cart items with product information
    cart_items = Cart.query.filter_by(user_id=session['user_id']).all()
    if not cart_items:
        flash('Your cart is empty', 'error')
        return redirect(url_for('cart'))
    
    # Prepare cart items with product details
    items_with_details = []
    total_price = 0
    
    for item in cart_items:
        product = Product.query.get(item.product_id)
        if product:
            item_details = {
                'name': product.name,
                'selected_size': item.size,
                'quantity': item.quantity,
                'price': item.price * item.quantity,
                'price_display': price_display(item.price * item.quantity)
            }
            items_with_details.append(item_details)
            total_price += item.price * item.quantity
    
    # Get user info for pre-filling
    user = User.query.get(session['user_id'])
    
    return render_template('order_details.html',
                          cart_items=items_with_details,
                          total_price=total_price,
                          total_price_display=price_display(total_price),
                          user=user)

@app.route('/my-orders')
def my_orders():
    if 'user_id' not in session:
        flash('Please login to view your orders', 'error')
        return redirect(url_for('login'))
    
    # Get all orders for the current user
    orders = Order.query.filter_by(user_id=session['user_id']).order_by(Order.created_at.desc()).all()
    return render_template('my_orders.html', orders=orders)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
