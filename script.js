let featureChart = null;
let importanceChart = null;

function createFeatureChart(data) {
    const ctx = document.getElementById('featureChart').getContext('2d');
    if (featureChart) {
        featureChart.destroy();
    }
    featureChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Overall Quality', 'Living Area', 'Basement', 'Garage Cars', 'Garage Area'],
            datasets: [{
                label: 'House Features',
                data: [
                    data.overallQual / 10,
                    data.grLivArea / 3000,
                    data.totalBsmtSF / 2000,
                    data.garageCars / 4,
                    data.garageArea / 1000
                ],
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });
}

document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = {
        overallQual: parseFloat(document.getElementById('overallQual').value),
        grLivArea: parseFloat(document.getElementById('grLivArea').value),
        totalBsmtSF: parseFloat(document.getElementById('totalBsmtSF').value),
        garageCars: parseFloat(document.getElementById('garageCars').value),
        garageArea: parseFloat(document.getElementById('garageArea').value)
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        const data = await response.json();
        
        document.getElementById('result').classList.remove('d-none');
        document.getElementById('predicted-price').textContent = 
            `$${data.predicted_price.toLocaleString()}`;
        
        // Create feature visualization
        createFeatureChart(formData);
        
        // Display confidence interval
        document.getElementById('confidenceInterval').textContent = 
            `Confidence Interval: $${data.confidence_interval[0].toLocaleString()} - $${data.confidence_interval[1].toLocaleString()}`;

    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while making the prediction.');
    }
});

// Initialize feature importance chart
fetch('/feature_importance')
    .then(response => response.json())
    .then(data => {
        const ctx = document.getElementById('importanceChart').getContext('2d');
        importanceChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.features,
                datasets: [{
                    label: 'Feature Importance',
                    data: data.importance,
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    });