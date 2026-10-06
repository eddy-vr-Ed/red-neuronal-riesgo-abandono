document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const btnSubmit = document.getElementById('btn-submit');
    const spinner = document.getElementById('spinner');

    // Result elements
    const statusCard = document.getElementById('status-card');
    const statusIcon = document.getElementById('status-icon');
    const riesgoLabel = document.getElementById('riesgo-label');
    const probBajoText = document.getElementById('prob-bajo-text');
    const probAltoText = document.getElementById('prob-alto-text');
    const barBajo = document.getElementById('bar-bajo');
    const barAlto = document.getElementById('bar-alto');

    // Modal elements
    const btnGraficas = document.getElementById('btn-graficas');
    const graficaModal = document.getElementById('grafica-modal');
    const closeModal = document.getElementById('close-modal');

    if (!form) return;

    // Modal Logic
    if (btnGraficas && graficaModal && closeModal) {
        btnGraficas.addEventListener('click', () => {
            graficaModal.classList.remove('hidden');
        });

        closeModal.addEventListener('click', () => {
            graficaModal.classList.add('hidden');
        });

        // Clic fuera del modal para cerrar
        window.addEventListener('click', (e) => {
            if (e.target === graficaModal) {
                graficaModal.classList.add('hidden');
            }
        });
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Show Loading
        btnSubmit.disabled = true;
        spinner.classList.remove('hidden');

        // Gather Data
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        try {
            // Fake delay for "Neural Network processing" dramatic effect (UX)
            await new Promise(r => setTimeout(r, 600));

            const response = await fetch('/predecir', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (response.ok) {
                // Update UI with results
                riesgoLabel.textContent = result.riesgo;

                probBajoText.textContent = result.probabilidad_bajo;
                probAltoText.textContent = result.probabilidad_alto;

                // Animate bars
                barBajo.style.width = result.probabilidad_bajo;
                barAlto.style.width = result.probabilidad_alto;

                // Update styling based on risk
                statusCard.className = 'status-card'; // reset
                if (result.riesgo === 'BAJO') {
                    statusCard.classList.add('status-BAJO');
                    statusIcon.textContent = '✓';
                } else {
                    statusCard.classList.add('status-ALTO');
                    statusIcon.textContent = '⚠';
                }

            } else {
                alert('Error de capa neuronal: ' + (result.error || 'Desconocido'));
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Error al conectar con la red neuronal.');
        } finally {
            // Hide loading
            btnSubmit.disabled = false;
            spinner.classList.add('hidden');
        }
    });
});
