const API_BASE_URL = 'http://localhost:8000';

document.getElementById('dataBtn').addEventListener('click', async () => {
    try {
        const response = await fetch(`${API_BASE_URL}/data/sample`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        displayData(data);
    } catch (error) {
        displayError('Failed to load market data: ' + error.message);
    }
});

document.getElementById('testBtn').addEventListener('click', async () => {
    const scenario = document.getElementById('testScenario').value;
    if (!scenario) {
        displayError('Please select a test scenario');
        return;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/test-vectors`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const vectors = await response.json();
        const selectedVector = vectors[scenario];

        if (!selectedVector) {
            displayError('Invalid scenario selected');
            return;
        }

        document.getElementById('sampleData').innerHTML = `
            <strong>${selectedVector.name}</strong><br>
            <small>${selectedVector.description}</small><br><br>
            <strong>State Vector (30 features):</strong><br>
            <code style="font-size: 11px;">${selectedVector.state.join(', ')}</code>
        `;

        // Send to backend for prediction
        console.log('Sending prediction request with state:', selectedVector.state);
        const predResponse = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ state: selectedVector.state }),
        });

        console.log('Prediction response status:', predResponse.status);

        if (!predResponse.ok) {
            const errorText = await predResponse.text();
            console.error('API Error:', errorText);
            throw new Error(`API Error ${predResponse.status}: ${errorText}`);
        }

        const result = await predResponse.json();
        console.log('Prediction result:', result);
        displayResult(result);
    } catch (error) {
        console.error('Test error:', error);
        displayError('Prediction failed: ' + error.message);
    }
});

function displayData(data) {
    const dataDisplay = document.getElementById('dataDisplay');
    dataDisplay.classList.remove('hidden');

    const newsDiv = document.getElementById('newsData');
    const sentimentDiv = document.getElementById('sentimentData');

    if (data.news_headlines && data.news_headlines.length > 0) {
        newsDiv.innerHTML = data.news_headlines.map(item =>
            `<div style="margin-bottom: 8px; padding: 8px; background: white; border-left: 3px solid #007bff;">
                <strong>${item.date}</strong> - ${item.category}<br>
                <em>${item.headline}</em>
            </div>`
        ).join('');
    }

    if (data.sentiment_analysis && data.sentiment_analysis.length > 0) {
        sentimentDiv.innerHTML = data.sentiment_analysis.map(item => {
            const score = parseFloat(item.sentiment_score);
            const color = score > 0.1 ? '#28a745' : score < -0.1 ? '#dc3545' : '#ffc107';
            return `<div style="margin-bottom: 8px; padding: 8px; background: white; border-left: 3px solid ${color};">
                <strong>${item.date}</strong> - ${item.ticker}<br>
                <em>${item.headline}</em><br>
                <small>Score: ${score.toFixed(3)}</small>
            </div>`;
        }).join('');
    }
}

function displayResult(result) {
    const resultDiv = document.getElementById('result');
    const contentDiv = document.getElementById('resultContent');

    resultDiv.classList.remove('hidden', 'error');

    // Handle both wrapped (result.prediction) and direct (result) formats
    const pred = result.prediction || result;

    if (pred.action_id !== undefined) {
        contentDiv.innerHTML = `
            <div style="line-height: 1.8;">
                <strong>Model:</strong> ${pred.model || 'sentiment'}<br>
                <strong>Action ID:</strong> ${pred.action_id}<br>
                <strong>Strategy:</strong> <span style="color: #007bff; font-weight: bold;">${pred.strategy}</span><br>
                <strong>Position:</strong> ${pred.position}<br>
                <strong>Holding Days:</strong> ${pred.holding_days}<br>
                <strong>Portfolio Signal:</strong> ${pred.portfolio_signal}<br>
                <strong>Expected Return:</strong> ${pred.expected_return_pct}%<br>
                <strong>Estimated Portfolio Value:</strong> $${pred.estimated_portfolio_value.toLocaleString()}<br>
                <strong>Profit/Loss Status:</strong> ${pred.profit_loss_status}
            </div>
        `;
    } else {
        displayError(result.message || 'Unknown error');
    }
}

function displayError(message) {
    const resultDiv = document.getElementById('result');
    const contentDiv = document.getElementById('resultContent');

    resultDiv.classList.remove('hidden');
    resultDiv.classList.add('error');
    contentDiv.innerHTML = `<strong>Error:</strong> ${message}`;
    console.error(message);
}