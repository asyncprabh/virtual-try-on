document.addEventListener('DOMContentLoaded', () => {
    const tryOnBtn = document.getElementById('tryOnBtn');
    const optionsContainer = document.getElementById('optionsContainer');
    const imageBtn = document.getElementById('imageBtn');
    const videoBtn = document.getElementById('videoBtn');
    const imageContainer = document.getElementById('imageContainer');
    const videoContainer = document.getElementById('videoContainer');
    const imageInput = document.getElementById('imageInput');
    const preview = document.getElementById('preview');
    const videoElement = document.getElementById('videoElement');

    // Show options when Try On button is clicked
    tryOnBtn.addEventListener('click', () => {
        optionsContainer.classList.remove('hidden');
    });

    // Handle image button click
    imageBtn.addEventListener('click', () => {
        imageContainer.classList.remove('hidden');
        videoContainer.classList.add('hidden');
        stopVideo();
    });

    // Handle video button click
    videoBtn.addEventListener('click', async () => {
        imageContainer.classList.add('hidden');
        videoContainer.classList.remove('hidden');
        startVideo();
    });

    // Handle image upload
    imageInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                preview.src = e.target.result;
                // Here you would typically call your virtual try-on AI model
                simulateVirtualTryOn('image');
            };
            reader.readAsDataURL(file);
        }
    });

    // Start video stream
    async function startVideo() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            videoElement.srcObject = stream;
            // Here you would typically call your virtual try-on AI model
            simulateVirtualTryOn('video');
        } catch (err) {
            console.error('Error accessing camera:', err);
            alert('Unable to access camera. Please make sure you have granted camera permissions.');
        }
    }

    // Stop video stream
    function stopVideo() {
        const stream = videoElement.srcObject;
        if (stream) {
            const tracks = stream.getTracks();
            tracks.forEach(track => track.stop());
            videoElement.srcObject = null;
        }
    }

    // Simulate virtual try-on (replace this with actual AI model integration)
    function simulateVirtualTryOn(mode) {
        const container = mode === 'image' ? 'virtualClothes' : 'virtualClothesVideo';
        const virtualClothes = document.getElementById(container);
        
        // This is a placeholder for the actual virtual try-on functionality
        // In a real implementation, you would integrate with an AI model
        virtualClothes.style.border = '2px dashed #4CAF50';
        virtualClothes.style.backgroundColor = 'rgba(76, 175, 80, 0.1)';
    }
});
