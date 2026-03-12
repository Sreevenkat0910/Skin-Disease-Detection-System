console.log('Detection.js loaded successfully');
let uploadedImage = null;

document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('uploadArea');
    const imageInput = document.getElementById('imageInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const previewImg = document.getElementById('previewImg');
    const imagePreview = document.getElementById('imagePreview');
    const uploadPrompt = document.getElementById('uploadPrompt');
    const removeImageBtn = document.getElementById('removeImage');
    const spinner = document.getElementById('spinner');

    // Click to upload
    uploadArea.addEventListener('click', function() {
        imageInput.click();
    });

    // Drag and drop functionality
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('border-blue-500', 'bg-blue-100');
    });

    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('border-blue-500', 'bg-blue-100');
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('border-blue-500', 'bg-blue-100');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleImageSelect(files[0]);
        }
    });

    // File input change
    imageInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            handleImageSelect(e.target.files[0]);
        }
    });

    // Remove image button
    removeImageBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        resetImageUpload();
    });

    // Analyze button click
    analyzeBtn.addEventListener('click', function() {
        if (uploadedImage) {
            analyzeSkinCondition();
        }
    });

    function handleImageSelect(file) {
        // Validate file type
        if (!file.type.match('image.*')) {
            alert('Please select a valid image file');
            return;
        }

        // Validate file size (max 16MB)
        if (file.size > 16 * 1024 * 1024) {
            alert('File size must be less than 16MB');
            return;
        }

        uploadedImage = file;
        
        // Preview the image
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImg.src = e.target.result;
            uploadPrompt.style.display = 'none';
            imagePreview.style.display = 'block';
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    function resetImageUpload() {
        uploadedImage = null;
        imageInput.value = '';
        uploadPrompt.style.display = 'block';
        imagePreview.style.display = 'none';
        analyzeBtn.disabled = true;
    }

    function analyzeSkinCondition() {
        const formData = new FormData();
        formData.append('image', uploadedImage);

        // Disable button and show loading state
        analyzeBtn.disabled = true;
        spinner.style.display = 'inline-block';
        analyzeBtn.innerHTML = '<span class="inline-block animate-spin mr-2">⏳</span>Analyzing...';

        // Send image to backend for analysis
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Store results in session storage
                sessionStorage.setItem('analysisResults', JSON.stringify(data));
                
                // Redirect to results page
                window.location.href = '/results';
            } else {
                alert('Error: ' + (data.error || 'Analysis failed'));
                resetAnalyzeButton();
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while analyzing the image. Please try again.');
            resetAnalyzeButton();
        });
    }

    function resetAnalyzeButton() {
        analyzeBtn.disabled = false;
        spinner.style.display = 'none';
        analyzeBtn.innerHTML = '<i class="bi bi-bar-chart-fill mr-2"></i>Analyze Skin Condition';
    }
});