const uploadButton = document.getElementById('upload');
const chosenImagePreview = document.getElementById('chosen-image-preview');
const heroImage = document.getElementById('chosen-image');
const heroPlaceholder = document.getElementById('hero-placeholder');
const previewCaption = document.getElementById('preview-caption');
const resultContainer = document.getElementById('result-container');
const errorContainer = document.getElementById('error-container');
const loadingPanel = document.getElementById('loading-spinner');

// Debug: log missing elements
if (!heroImage) console.warn('Missing #chosen-image');
if (!heroPlaceholder) console.warn('Missing #hero-placeholder');
if (!chosenImagePreview) console.warn('Missing #chosen-image-preview');
if (!previewCaption) console.warn('Missing #preview-caption');

function resetPanels() {
  if (resultContainer) {
    resultContainer.innerHTML = '';
    resultContainer.classList.remove('active');
  }
  if (errorContainer) {
    errorContainer.innerHTML = '';
    errorContainer.classList.remove('active');
  }
  if (loadingPanel) {
    loadingPanel.classList.remove('active');
  }
}

function resetPreview() {
  if (heroImage) heroImage.style.display = 'none';
  if (chosenImagePreview) chosenImagePreview.style.display = 'none';
  if (heroPlaceholder) heroPlaceholder.style.display = 'flex';
  if (previewCaption) previewCaption.style.display = 'none';
}

// Only initialize if DOM is ready
if (uploadButton && resultContainer && errorContainer && loadingPanel) {
  resetPreview();
  resetPanels();
}

if (uploadButton) {
  uploadButton.addEventListener('change', () => {
  const file = uploadButton.files[0];
  if (!file) {
    resetPreview();
    return;
  }

  const reader = new FileReader();
  reader.readAsDataURL(file);
  reader.onload = () => {
    if (chosenImagePreview) chosenImagePreview.src = reader.result;
    if (heroImage) heroImage.src = reader.result;
    if (chosenImagePreview) chosenImagePreview.style.display = 'block';
    if (heroImage) heroImage.style.display = 'block';
    if (heroPlaceholder) heroPlaceholder.style.display = 'none';
    if (previewCaption) previewCaption.style.display = 'block';
  };
});
}

function showLoadingSpinner() {
  if (loadingPanel) loadingPanel.classList.add('active');
}

function hideLoadingSpinner() {
  if (loadingPanel) loadingPanel.classList.remove('active');
}

function displayResponse(response) {
  resetPanels();
  if (!resultContainer) return;
  resultContainer.classList.add('active');

  const diseaseBlock = document.createElement('div');
  diseaseBlock.className = 'result-item';
  diseaseBlock.innerHTML = `<strong>Disease</strong><span>${response.disease}</span>`;

  const accuracyBlock = document.createElement('div');
  accuracyBlock.className = 'result-item';
  accuracyBlock.innerHTML = `<strong>Confidence</strong><span>${response.accuracy}%</span>`;

  const medicineBlock = document.createElement('div');
  medicineBlock.className = 'result-item';
  medicineBlock.innerHTML = `<strong>Recommended action</strong><span>${response.medicine}</span>`;

  const detectedBlock = document.createElement('div');
  detectedBlock.className = 'result-item';
  detectedBlock.innerHTML = `<strong>Detected</strong><span>${response.detected ? 'Yes' : 'No'}</span>`;

  resultContainer.appendChild(diseaseBlock);
  resultContainer.appendChild(accuracyBlock);
  resultContainer.appendChild(medicineBlock);
  resultContainer.appendChild(detectedBlock);
}

function displayError(message) {
  resetPanels();
  if (!errorContainer) return;
  errorContainer.classList.add('active');
  errorContainer.textContent = message;
}

function detectFunction() {
  const fileInput = document.getElementById('upload');
  if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
    displayError('Please choose an image before clicking Analyze Image.');
    return;
  }

  console.log('Starting detection with file:', fileInput.files[0].name, 'Size:', fileInput.files[0].size);
  resetPanels();
  showLoadingSpinner();

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);

  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/detect', true);
  
  // Timeout after 60 seconds
  xhr.timeout = 60000;
  
  xhr.onload = function () {
    console.log('Response received. Status:', xhr.status);
    hideLoadingSpinner();
    if (xhr.status === 200) {
      const response = JSON.parse(xhr.responseText);
      console.log('Parsed response:', response);
      displayResponse(response);
    } else {
      console.error('Error response:', xhr.responseText);
      let response = {};
      try {
        response = JSON.parse(xhr.responseText);
      } catch (err) {
        response.message = 'Unable to analyze the image. Please try again.';
      }
      displayError(response.message ? 'Error: ' + response.message : 'Unable to analyze the image.');
    }
  };
  
  xhr.onerror = function () {
    console.error('Network error occurred');
    hideLoadingSpinner();
    displayError('A network error occurred. Please try again.');
  };
  
  xhr.ontimeout = function () {
    console.error('Request timeout after 60 seconds');
    hideLoadingSpinner();
    displayError('Request timed out. The server took too long to respond. Please try again.');
  };
  
  console.log('Sending FormData with file to /detect endpoint');
  xhr.send(formData);
}
