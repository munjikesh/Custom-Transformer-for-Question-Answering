document.addEventListener('DOMContentLoaded', () => {
    const qaForm = document.getElementById('qa-form');
    const modelTypeSelect = document.getElementById('model-type');
    const extSettings = document.getElementById('extractive-settings');
    const genSettings = document.getElementById('generative-settings');
    const accordionHeader = document.getElementById('advanced-settings-toggle');
    const resultContainer = document.getElementById('result-container');
    const loader = document.getElementById('loader');
    const resultContent = document.getElementById('result-content');
    const answerText = document.getElementById('answer-text');
    const statsContainer = document.getElementById('stats-container');
    const submitBtn = document.getElementById('submit-btn');

    let abortController = null;

    // Toggle Advanced Settings Accordion
    accordionHeader.addEventListener('click', () => {
        accordionHeader.classList.toggle('active');
    });

    // Update range input displays
    const ranges = document.querySelectorAll('input[type="range"]');
    ranges.forEach(range => {
        const display = document.getElementById(`${range.id}-val`);
        range.addEventListener('input', () => {
            display.textContent = range.value;
        });
    });

    // Toggle settings based on model type
    modelTypeSelect.addEventListener('change', (e) => {
        if (e.target.value === 'extractive') {
            extSettings.classList.remove('hidden');
            genSettings.classList.add('hidden');
        } else {
            extSettings.classList.add('hidden');
            genSettings.classList.remove('hidden');
        }
    });

    // Handle enable no-answer gate checkbox
    const enableGateCheckbox = document.getElementById('enable-no-answer-gate');
    const gateThresholdContainer = document.getElementById('gate-threshold-container');
    enableGateCheckbox.addEventListener('change', (e) => {
        if (e.target.checked) {
            gateThresholdContainer.classList.remove('hidden');
        } else {
            gateThresholdContainer.classList.add('hidden');
        }
    });

    // Submit Handler
    qaForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        if (abortController) {
            abortController.abort(); // Cancel previous request
        }
        abortController = new AbortController();

        const modelType = modelTypeSelect.value;
        const payload = {
            model_type: modelType,
            context: document.getElementById('context').value,
            question: document.getElementById('question').value,
            max_length: parseInt(document.getElementById('max-length').value),
        };

        if (modelType === 'extractive') {
            payload.doc_stride = parseInt(document.getElementById('doc-stride').value);
            payload.n_best = parseInt(document.getElementById('n-best').value);
            payload.max_answer_length = parseInt(document.getElementById('max-answer-length').value);
            payload.no_answer_threshold = parseFloat(document.getElementById('ext-no-answer-threshold').value);
        } else {
            payload.beam_size = parseInt(document.getElementById('beam-size').value);
            payload.max_new_tokens = parseInt(document.getElementById('max-new-tokens').value);
            payload.length_penalty = parseFloat(document.getElementById('length-penalty').value);
            payload.enable_no_answer_gate = enableGateCheckbox.checked;
            payload.no_answer_threshold = parseFloat(document.getElementById('gen-no-answer-threshold').value);
        }

        // UI Reset for loading
        resultContainer.classList.remove('hidden');
        resultContent.classList.add('hidden');
        loader.classList.remove('hidden');
        submitBtn.disabled = true;
        submitBtn.textContent = "Processing...";

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
                signal: abortController.signal
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || "Server error");
            }

            const data = await response.json();
            
            // Display Results
            loader.classList.add('hidden');
            resultContent.classList.remove('hidden');
            
            answerText.textContent = data.answer || "[No Answer / Unanswerable]";
            
            // Build stats
            let statsHtml = '';
            if (modelType === 'extractive') {
                statsHtml += `<span>Span Score: ${data.span_score.toFixed(4)}</span>`;
                statsHtml += `<span>Null Score: ${data.null_score.toFixed(4)}</span>`;
                statsHtml += `<span>Score Diff (Null - Span): ${data.score_diff_null_minus_span.toFixed(4)}</span>`;
                statsHtml += `<span>Predicted No-Answer: ${data.predicted_no_answer ? "Yes" : "No"}</span>`;
            } else {
                if (data.gate && data.gate.enabled !== false) {
                    statsHtml += `<span>Score Diff: ${data.gate.score_diff.toFixed(4)}</span>`;
                    statsHtml += `<span>Predicted No-Answer: ${data.gate.selected_no_answer ? "Yes" : "No"}</span>`;
                }
            }
            statsContainer.innerHTML = statsHtml;

        } catch (error) {
            if (error.name === 'AbortError') {
                console.log('Request cancelled by new submission');
                return; // Don't reset UI, a new request is taking over
            }
            loader.classList.add('hidden');
            resultContent.classList.remove('hidden');
            answerText.textContent = `Error: ${error.message}`;
            statsContainer.innerHTML = '';
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = "Generate Answer";
        }
    });
});
