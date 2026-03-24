let chartInstance = null;

document.addEventListener("DOMContentLoaded", () => {
    loadMetrics();
    fetchPrediction(); // Initial load
    
    document.getElementById("predict-btn").addEventListener("click", () => {
        fetchPrediction();
    });
});

async function fetchPrediction() {
    const model = document.getElementById("model-select").value;
    const days = document.getElementById("days-input").value;
    const spinner = document.getElementById("loading-spinner");
    
    spinner.classList.remove("hidden");
    
    try {
        const response = await fetch(`/api/predict?model=${model}&days=${days}`);
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Failed to fetch prediction");
        }
        const data = await response.json();
        renderChart(data);
    } catch (error) {
        alert("Error: " + error.message);
    } finally {
        spinner.classList.add("hidden");
    }
}

async function loadMetrics() {
    try {
        const response = await fetch("/api/metrics");
        const data = await response.json();
        
        const tbody = document.querySelector("#metrics-table tbody");
        tbody.innerHTML = "";
        
        if (data.message) {
            tbody.innerHTML = `<tr><td colspan="5">${data.message}</td></tr>`;
            return;
        }
        
        for (const [modelName, metrics] of Object.entries(data)) {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><strong>${modelName}</strong></td>
                <td>${metrics.mae.toFixed(2)}</td>
                <td>${metrics.rmse.toFixed(2)}</td>
                <td>${metrics.mape.toFixed(2)}%</td>
                <td>${metrics.mase.toFixed(4)}</td>
            `;
            tbody.appendChild(tr);
        }
    } catch (error) {
        console.error("Failed to load metrics:", error);
    }
}

function renderChart(data) {
    const ctx = document.getElementById('forecastChart').getContext('2d');
    
    if (chartInstance) {
        chartInstance.destroy();
    }
    
    // Combine datasets for x-axis
    const allLabels = [...data.historical_dates, ...data.future_dates];
    
    // Create corresponding null-padded arrays for charting
    const histData = [...data.historical_prices, ...Array(data.future_dates.length).fill(null)];
    
    // To connect the line smoothly, repeat the last historical point in the future array
    const lastHistPrice = data.historical_prices[data.historical_prices.length - 1];
    
    const futureDataArray = Array(data.historical_dates.length - 1).fill(null);
    futureDataArray.push(lastHistPrice);
    futureDataArray.push(...data.future_prices);
    
    const lowerBoundArray = Array(data.historical_dates.length - 1).fill(null);
    lowerBoundArray.push(lastHistPrice);
    lowerBoundArray.push(...data.lower_bounds);
    
    const upperBoundArray = Array(data.historical_dates.length - 1).fill(null);
    upperBoundArray.push(lastHistPrice);
    upperBoundArray.push(...data.upper_bounds);

    chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: allLabels,
            datasets: [
                {
                    label: 'Historical Price',
                    data: histData,
                    borderColor: '#79c0ff',
                    backgroundColor: 'rgba(121, 192, 255, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    fill: true,
                    tension: 0.1
                },
                {
                    label: `Forecast (${data.model_name})`,
                    data: futureDataArray,
                    borderColor: '#ff7b72',
                    borderWidth: 3,
                    borderDash: [5, 5],
                    pointRadius: 3,
                    pointBackgroundColor: '#ff7b72',
                    fill: false,
                    tension: 0.1
                },
                {
                    label: 'Upper Bound',
                    data: upperBoundArray,
                    borderColor: 'rgba(255, 123, 114, 0)',
                    backgroundColor: 'rgba(255, 123, 114, 0.1)',
                    fill: '+1', // Fill to next dataset (lower bound)
                    pointRadius: 0,
                    tension: 0.1
                },
                {
                    label: 'Lower Bound',
                    data: lowerBoundArray,
                    borderColor: 'rgba(255, 123, 114, 0)',
                    backgroundColor: 'transparent',
                    fill: false,
                    pointRadius: 0,
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    labels: { color: '#c9d1d9', font: { family: 'Outfit', size: 13 } }
                },
                tooltip: {
                    backgroundColor: 'rgba(13, 17, 23, 0.9)',
                    titleColor: '#fff',
                    bodyColor: '#c9d1d9',
                    borderColor: '#30363d',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: true
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(48, 54, 61, 0.5)' },
                    ticks: { color: '#8b949e', maxTicksLimit: 12 }
                },
                y: {
                    grid: { color: 'rgba(48, 54, 61, 0.5)' },
                    ticks: {
                        color: '#8b949e',
                        callback: function(value) { return '$' + value; }
                    }
                }
            }
        }
    });
}
