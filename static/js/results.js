console.log('Results.js loaded successfully');

document.addEventListener('DOMContentLoaded', function() {
    const results = JSON.parse(sessionStorage.getItem('analysisResults'));
    
    if (!results) {
        window.location.href = '/detection';
        return;
    }

    displayResults(results);
});

function displayResults(results) {
    // Display skin condition results
    document.getElementById('resultImage').src = `/static/uploads/${results.image_path}`;
    document.getElementById('skinCondition').textContent = results.skin_condition;
    
    const confidence = results.skin_confidence;
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceText = document.getElementById('confidenceText');
    
    // Animate confidence bar
    setTimeout(() => {
        confidenceBar.style.width = confidence + '%';
    }, 300);
    
    confidenceText.textContent = confidence.toFixed(2) + '%';
    
    // Set confidence bar color based on confidence level
    confidenceBar.classList.remove('from-green-400', 'to-green-600', 'from-yellow-400', 'to-yellow-600', 'from-red-400', 'to-red-600');
    
    if (confidence >= 80) {
        confidenceBar.classList.add('from-green-400', 'to-green-600');
        confidenceText.classList.add('text-green-600');
    } else if (confidence >= 60) {
        confidenceBar.classList.add('from-yellow-400', 'to-yellow-600');
        confidenceText.classList.remove('text-green-600');
        confidenceText.classList.add('text-yellow-600');
    } else {
        confidenceBar.classList.add('from-red-400', 'to-red-600');
        confidenceText.classList.remove('text-green-600');
        confidenceText.classList.add('text-red-600');
    }
    
    // Display all predictions if available
    if (results.all_predictions && results.all_predictions.length > 0) {
        displayAllPredictions(results.all_predictions);
    }
    
    // Display recommendations with delay
    setTimeout(() => {
        displayRecommendations(results.recommendations);
    }, 2000);
}

function displayAllPredictions(predictions) {
    const allPredictionsContainer = document.getElementById('allPredictions');
    allPredictionsContainer.innerHTML = '';
    
    predictions.forEach((prediction, index) => {
        const predictionDiv = document.createElement('div');
        predictionDiv.className = 'flex justify-between items-center p-4 bg-gray-50 rounded-xl hover:bg-gray-100 transition-all';
        
        const labelDiv = document.createElement('div');
        labelDiv.className = 'font-semibold text-gray-800';
        labelDiv.textContent = `${index + 1}. ${prediction.class}`;
        
        const confidenceDiv = document.createElement('div');
        confidenceDiv.className = 'flex items-center gap-3';
        
        const confidenceBar = document.createElement('div');
        confidenceBar.className = 'w-32 h-3 bg-gray-200 rounded-full overflow-hidden';
        
        const confidenceFill = document.createElement('div');
        confidenceFill.className = 'h-full bg-gradient-to-r from-blue-400 to-cyan-500 transition-all duration-500';
        confidenceFill.style.width = '0%';
        
        confidenceBar.appendChild(confidenceFill);
        
        const confidenceText = document.createElement('span');
        confidenceText.className = 'font-bold text-gray-700 text-sm';
        confidenceText.textContent = `${prediction.confidence.toFixed(2)}%`;
        
        confidenceDiv.appendChild(confidenceBar);
        confidenceDiv.appendChild(confidenceText);
        
        predictionDiv.appendChild(labelDiv);
        predictionDiv.appendChild(confidenceDiv);
        
        allPredictionsContainer.appendChild(predictionDiv);
        
        // Animate the confidence bar
        setTimeout(() => {
            confidenceFill.style.width = prediction.confidence + '%';
        }, 300 + (index * 100));
    });
}

function displayRecommendations(recommendations) {
    document.getElementById('loadingRecommendations').style.display = 'none';
    document.getElementById('recommendations').style.display = 'block';
    
    const formattedRecommendations = formatMarkdown(recommendations);
    document.getElementById('recommendationsContent').innerHTML = formattedRecommendations;
}

function formatMarkdown(text) {
    // Convert headers with Tailwind styling
    text = text.replace(/^### (.*$)/gim, '<h3 class="text-xl font-bold text-gray-800 mt-6 mb-3">$1</h3>');
    text = text.replace(/^## (.*$)/gim, '<h2 class="text-2xl font-bold text-gray-800 mt-6 mb-4">$1</h2>');
    text = text.replace(/^# (.*$)/gim, '<h1 class="text-3xl font-bold text-gray-800 mt-6 mb-4">$1</h1>');
    
    // Convert bold and italic
    text = text.replace(/\*\*\*(.+?)\*\*\*/g, '<strong class="font-bold text-gray-900"><em>$1</em></strong>');
    text = text.replace(/\*\*(.+?)\*\*/g, '<strong class="font-bold text-gray-900">$1</strong>');
    text = text.replace(/\*(.+?)\*/g, '<em class="italic">$1</em>');
    
    // Convert lists
    text = text.replace(/^\* (.+)$/gim, '<li class="ml-6 mb-2">$1</li>');
    text = text.replace(/^- (.+)$/gim, '<li class="ml-6 mb-2">$1</li>');
    text = text.replace(/^(\d+)\. (.+)$/gim, '<li class="ml-6 mb-2">$2</li>');
    
    // Wrap lists in ul tags with styling
    text = text.replace(/(<li.*?<\/li>)/s, '<ul class="list-disc list-inside space-y-2 my-4">$1</ul>');
    
    // Convert line breaks to paragraphs with styling
    text = text.replace(/\n\n/g, '</p><p class="mb-4 text-gray-700 leading-relaxed">');
    text = '<p class="mb-4 text-gray-700 leading-relaxed">' + text + '</p>';
    
    return text;
}