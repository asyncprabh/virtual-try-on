document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const captureBtn = document.getElementById('capture-btn');
    const uploadBtn = document.getElementById('upload-btn');
    const imageUpload = document.getElementById('image-upload');
    const tryOnModal = document.getElementById('try-on-modal');
    const closeModal = document.querySelector('.close');
    const tryOnResult = document.getElementById('try-on-result');

    // Initialize camera
    async function initCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
        } catch (err) {
            console.error('Error accessing camera:', err);
            alert('Error accessing camera. Please make sure you have granted camera permissions.');
        }
    }

    // Initialize camera when page loads
    initCamera();

    // Capture image from camera
    captureBtn.addEventListener('click', function() {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        
        canvas.toBlob(function(blob) {
            processMeasurements(blob);
        }, 'image/jpeg');
    });

    // Handle image upload
    uploadBtn.addEventListener('click', function() {
        imageUpload.click();
    });

    imageUpload.addEventListener('change', function(e) {
        if (e.target.files && e.target.files[0]) {
            processMeasurements(e.target.files[0]);
        }
    });

    // Process measurements
    async function processMeasurements(imageBlob) {
        const formData = new FormData();
        formData.append('image', imageBlob);

        try {
            const response = await fetch('/api/measurements', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            if (response.ok) {
                updateMeasurements(data);
            } else {
                alert(data.error || 'Error processing measurements');
            }
        } catch (err) {
            console.error('Error:', err);
            alert('Error processing measurements');
        }
    }

    // Update measurements display
    function updateMeasurements(data) {
        document.getElementById('height-value').textContent = `${Math.round(data.height)} cm`;
        document.getElementById('shoulder-width-value').textContent = `${Math.round(data.shoulder_width)} cm`;
    }

    // Try-on functionality
    document.querySelectorAll('.try-on-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const productId = this.closest('.product-card').dataset.productId;
            const productImage = this.closest('.product-card').querySelector('img').src;
            
            tryOnResult.src = productImage; // Temporary: Just showing product image
            tryOnModal.style.display = 'block';
        });
    });

    // Modal controls
    closeModal.addEventListener('click', function() {
        tryOnModal.style.display = 'none';
    });

    window.addEventListener('click', function(e) {
        if (e.target == tryOnModal) {
            tryOnModal.style.display = 'none';
        }
    });

    // Add to cart functionality
    document.querySelectorAll('.add-to-cart-btn, #add-to-cart-modal').forEach(btn => {
        btn.addEventListener('click', async function() {
            const productId = this.closest('.product-card')?.dataset.productId;
            const size = document.getElementById('size')?.value || 'M';

            try {
                const response = await fetch('/api/cart/add', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        product_id: productId,
                        size: size,
                        quantity: 1
                    })
                });

                const data = await response.json();
                if (response.ok) {
                    alert('Product added to cart!');
                    if (tryOnModal.style.display === 'block') {
                        tryOnModal.style.display = 'none';
                    }
                } else {
                    alert(data.error || 'Error adding to cart');
                }
            } catch (err) {
                console.error('Error:', err);
                alert('Error adding to cart');
            }
        });
    });
});
