from .database import db

class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    price = db.Column(db.Float, nullable=False)
    size_chart = db.Column(db.JSON)
    image_url = db.Column(db.String(200))
    category = db.Column(db.String(50))
    try_on_enabled = db.Column(db.Boolean, default=False)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'price': self.price,
            'size_chart': self.size_chart,
            'image_url': self.image_url,
            'category': self.category,
            'try_on_enabled': self.try_on_enabled
        }

    @staticmethod
    def get_sample_products():
        return [
            {
                'name': 'Classic White T-Shirt',
                'description': 'A premium cotton t-shirt perfect for everyday wear',
                'price': 29.99,
                'size_chart': {'S': [36, 28], 'M': [38, 30], 'L': [40, 32], 'XL': [42, 34]},
                'image_url': 'images/19620598_44329692_1000.jpg',
                'category': 'T-Shirts',
                'try_on_enabled': True
            },
            {
                'name': 'Polo Shirt',
                'description': 'Classic polo shirt with comfortable fit',
                'price': 39.99,
                'size_chart': {'S': [36, 28], 'M': [38, 30], 'L': [40, 32], 'XL': [42, 34]},
                'image_url': 'images/2.png',
                'category': 'T-Shirts',
                'try_on_enabled': True
            },
            {
                'name': 'Denim Jacket',
                'description': 'Stylish denim jacket with a modern fit',
                'price': 79.99,
                'size_chart': {'S': [36, 24], 'M': [38, 25], 'L': [40, 26], 'XL': [42, 27]},
                'image_url': 'images/7.png',
                'category': 'Jackets',
                'try_on_enabled': True
            },
            {
                'name': 'Formal Black Blazer',
                'description': 'Elegant black blazer for formal occasions',
                'price': 129.99,
                'size_chart': {'S': [36, 28], 'M': [38, 29], 'L': [40, 30], 'XL': [42, 31]},
                'image_url': 'images/R.jpeg',
                'category': 'Formal Wear',
                'try_on_enabled': True
            },
            {
                'name': 'Casual Hoodie',
                'description': 'Comfortable hoodie for casual wear',
                'price': 49.99,
                'size_chart': {'S': [36, 26], 'M': [38, 27], 'L': [40, 28], 'XL': [42, 29]},
                'image_url': 'images/6.png',
                'category': 'Casual Wear',
                'try_on_enabled': True
            }
        ]
