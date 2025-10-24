# Virtual Try-On System Setup

## Directory Structure
```
virtual-tryon/
├── static/
│   ├── products/        # Store product images here
│   │   ├── tshirt.png   # Sample product (must be PNG with transparency)
│   │   └── jacket.png   # Sample product (must be PNG with transparency)
│   └── uploads/         # User uploaded photos and results
├── templates/
└── ...
```

## Product Images Requirements
1. Must be PNG format with transparency
2. Background should be transparent
3. Clothing should be front-facing
4. Image should be high quality and well-lit
5. Recommended size: 800x800 pixels minimum

## Setting Up Products
1. Place product images in `static/products/` directory
2. Make sure image filenames match the database entries
3. Product images should be referenced in database as 'products/filename.png'

## Common Issues
1. If clothing not showing up:
   - Check if product image exists in correct location
   - Ensure image has transparency
   - Verify database image_url is correct
   - Check file permissions

2. If clothing position is wrong:
   - Ensure person is standing straight
   - Full upper body should be visible
   - Good lighting is important
   - Solid background works best
