/**
 * Lógica del Cliente (Frontend) - Red Neuronal v2.0
 * Nivel: Técnico Superior Universitario (TSU) en Desarrollo de Software
 * 
 * Este archivo gestiona:
 * 1. La interacción con los controles de selección (grado, grupo, horas).
 * 2. La validación, casteo y envío de datos vía fetch() a Flask (/predecir).
 * 3. La actualización reactiva del DOM con los resultados de la inferencia.
 * 4. El procesamiento por lotes con archivos Excel/CSV (/predecir_lote).
 */

document.addEventListener('DOMContentLoaded', () => {

    // =========================================================================
    // 1. REFERENCIAS A ELEMENTOS DEL DOM (FORMULARIO Y CONTROLES)
    // =========================================================================
    const form = document.getElementById('prediction-form');
    const btnSubmit = document.getElementById('btn-submit');
    const spinner = document.getElementById('spinner');

    const gradoSelect = document.getElementById('grado');
    const grupoSelect = document.getElementById('grupo');
    const especialidadGroup = document.getElementById('especialidad-group');
    const especialidadSelect = document.getElementById('especialidad');
    const horasInput = document.getElementById('horas_semana_totales');
    const asistenciaInput = document.getElementById('asistencia_semanal');
    const promedioInput = document.getElementById('promedio');
    const reprobadasInput = document.getElementById('materias_reprobadas');

    // Elementos de la tarjeta de resultados
    const statusCard = document.getElementById('status-card');
    const statusIcon = document.getElementById('status-icon');
    const riesgoLabel = document.getElementById('riesgo-label');
    const riesgoDesc = document.getElementById('riesgo-desc');
    const probBajoText = document.getElementById('prob-bajo-text');
    const probAltoText = document.getElementById('prob-alto-text');
    const barBajo = document.getElementById('bar-bajo');
    const barAlto = document.getElementById('bar-alto');

    // Modal de curvas de aprendizaje
    const btnGraficas = document.getElementById('btn-graficas');
    const graficaModal = document.getElementById('grafica-modal');
    const closeModal = document.getElementById('close-modal');

    // =========================================================================
    // 2. CONFIGURACIÓN INSTITUCIONAL DE GRUPOS Y CARGA HORARIA
    // =========================================================================
    const configuracionAcademica = {
        "1": { grupos: ["A", "B", "C", "D"], horas: 35 },
        "4": { grupos: ["A", "B", "C"], horas: 35 },
        "7": { grupos: ["A", "B", "C"], horas: 35 },
        "10": { grupos: ["A", "B"], horas: 25 }
    };

    // Evento: Al cambiar el cuatrimestre seleccionado
    gradoSelect.addEventListener('change', () => {
        const grado = gradoSelect.value;
        const config = configuracionAcademica[grado];

        if (!config) return;

        // Reiniciar y poblar el combo de grupos
        grupoSelect.innerHTML = '<option value="" disabled selected>Seleccione grupo...</option>';
        grupoSelect.disabled = false;

        config.grupos.forEach(letra => {
            const opcion = document.createElement('option');
            opcion.value = letra;
            opcion.textContent = `Grupo ${letra}`;
            grupoSelect.appendChild(opcion);
        });

        // Mostrar u ocultar el campo de especialidad (exclusivo para 10mo)
        if (grado === "10") {
            especialidadGroup.style.display = 'block';
            especialidadSelect.disabled = false;
            horasInput.value = config.horas;
        } else {
            especialidadGroup.style.display = 'none';
            especialidadSelect.disabled = true;
            horasInput.value = config.horas;
        }
    });

    // Evento: Al cambiar de grupo
    grupoSelect.addEventListener('change', () => {
        const grado = gradoSelect.value;
        if (grado === "10") {
            validarEspecialidad10mo();
        } else if (configuracionAcademica[grado]) {
            horasInput.value = configuracionAcademica[grado].horas;
        }
    });

    // Evento: Al cambiar especialidad en 10mo
    especialidadSelect.addEventListener('change', () => {
        if (gradoSelect.value === "10") {
            validarEspecialidad10mo();
        }
    });

    function validarEspecialidad10mo() {
        const especialidad = especialidadSelect.value;
        const grupo = grupoSelect.value;

        // Regla institucional: Redes únicamente cuenta con Grupo A
        if (especialidad === "redes" && grupo === "B") {
            alert("Nota: La especialidad de Redes solo cuenta con el Grupo A.");
            grupoSelect.value = "A";
        }
        horasInput.value = 25;
    }

    // =========================================================================
    // 3. ENVÍO DE DATOS A LA RED NEURONAL (INFERENCIA INDIVIDUAL)
    // =========================================================================
    form.addEventListener('submit', async (evento) => {
        // Evitamos que el formulario recargue la página completa
        evento.preventDefault();

        // Validación de campos requeridos
        if (!gradoSelect.value) {
            alert("Por favor selecciona un cuatrimestre.");
            gradoSelect.focus();
            return;
        }
        if (!grupoSelect.value) {
            alert("Por favor selecciona un grupo.");
            grupoSelect.focus();
            return;
        }

        // Extracción explícita y casteo numérico de cada campo
        const grado = parseInt(gradoSelect.value, 10);
        const grupo = grupoSelect.value.trim().toUpperCase();
        const especialidad = (grado === 10) ? especialidadSelect.value.trim().toLowerCase() : "general";
        const horasSemana = parseFloat(horasInput.value) || 35.0;
        const asistencia = parseFloat(asistenciaInput.value);
        const promedio = parseFloat(promedioInput.value);
        const reprobadas = parseInt(reprobadasInput.value, 10) || 0;

        // Validaciones numéricas de rango
        if (isNaN(asistencia) || asistencia < 0 || asistencia > 100) {
            alert("La asistencia debe ser un número válido entre 0 y 100%.");
            asistenciaInput.focus();
            return;
        }
        if (isNaN(promedio) || promedio < 0 || promedio > 10) {
            alert("El promedio debe ser un número válido entre 0 y 10.");
            promedioInput.focus();
            return;
        }

        // Objeto JSON limpio y tipado para enviar a Flask
        const payloadEstudiante = {
            grado: grado,
            grupo: grupo,
            especialidad: especialidad,
            horas_semana_totales: horasSemana,
            asistencia_semanal: asistencia,
            promedio: promedio,
            materias_reprobadas: reprobadas
        };

        // Feedback visual: deshabilitar botón y mostrar spinner
        btnSubmit.disabled = true;
        spinner.classList.remove('hidden');

        try {
            console.log("-> Enviando datos a /predecir:", payloadEstudiante);

            // LLAMADA HTTP POST CON FETCH
            const respuesta = await fetch('/predecir', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payloadEstudiante)
            });

            const resultado = await respuesta.json();
            console.log("<- Respuesta recibida de Flask:", resultado);

            if (!respuesta.ok) {
                alert("Error del modelo: " + (resultado.error || "Ocurrió un error al procesar la predicción."));
                return;
            }

            // ACTUALIZACIÓN DE LA INTERFAZ CON EL RESULTADO DE LA RED
            riesgoLabel.textContent = `RIESGO ${resultado.riesgo}`;
            probBajoText.textContent = resultado.probabilidad_bajo;
            probAltoText.textContent = resultado.probabilidad_alto;

            barBajo.style.width = resultado.probabilidad_bajo;
            barAlto.style.width = resultado.probabilidad_alto;

            // Actualizar estilo de la tarjeta de veredicto
            statusCard.className = 'verdict-card';
            if (resultado.riesgo === 'BAJO') {
                statusCard.classList.add('status-BAJO');
                statusIcon.textContent = '✓';
                if (riesgoDesc) {
                    riesgoDesc.textContent = 'Trayectoria regular: probabilidad favorable de permanencia y retención escolar.';
                }
            } else {
                statusCard.classList.add('status-ALTO');
                statusIcon.textContent = '⚠';
                if (riesgoDesc) {
                    riesgoDesc.textContent = 'Alerta crítica: probabilidad de abandono elevada. Requiere canalización y tutoría inmediata.';
                }
            }

            // Actualizar chips de pesos sinápticos visibles
            if (Array.isArray(resultado.pesos_ejemplo)) {
                const chipsVal = document.querySelectorAll('.chip-val');
                resultado.pesos_ejemplo.forEach((peso, index) => {
                    if (chipsVal[index]) {
                        chipsVal[index].textContent = Number(peso).toFixed(4);
                    }
                });
            }

        } catch (error) {
            console.error("Error en la petición fetch:", error);
            alert("No se pudo conectar con el servidor Flask. Verifica que la terminal de python app.py esté en ejecución.");
        } finally {
            // Restaurar estado del botón
            btnSubmit.disabled = false;
            spinner.classList.add('hidden');
        }
    });

    // =========================================================================
    // 4. CONTROL DE PESTAÑAS (CONSULTA INDIVIDUAL VS ANÁLISIS POR LOTES)
    // =========================================================================
    const tabBtnIndividual = document.getElementById('tab-btn-individual');
    const tabBtnLotes = document.getElementById('tab-btn-lotes');
    const viewIndividual = document.getElementById('view-individual');
    const viewLotes = document.getElementById('view-lotes');

    if (tabBtnIndividual && tabBtnLotes && viewIndividual && viewLotes) {
        tabBtnIndividual.addEventListener('click', () => {
            tabBtnIndividual.classList.add('active');
            tabBtnLotes.classList.remove('active');
            viewIndividual.classList.remove('hidden');
            viewLotes.classList.add('hidden');
        });

        tabBtnLotes.addEventListener('click', () => {
            tabBtnLotes.classList.add('active');
            tabBtnIndividual.classList.remove('active');
            viewLotes.classList.remove('hidden');
            viewIndividual.classList.add('hidden');
        });
    }

    // =========================================================================
    // 5. PROCESAMIENTO POR LOTES CON EXCEL / CSV
    // =========================================================================
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const selectedFileInfo = document.getElementById('selected-file-info');
    const fileNameText = document.getElementById('file-name-text');
    const btnRemoveFile = document.getElementById('btn-remove-file');
    const btnProcessBatch = document.getElementById('btn-process-batch');
    const batchSpinner = document.getElementById('batch-spinner');
    const batchResults = document.getElementById('batch-results');

    let archivoSeleccionado = null;

    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                seleccionarArchivo(e.target.files[0]);
            }
        });
    }

    if (uploadZone) {
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        });
        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('dragover');
        });
        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                seleccionarArchivo(e.dataTransfer.files[0]);
            }
        });
    }

    function seleccionarArchivo(file) {
        archivoSeleccionado = file;
        fileNameText.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        selectedFileInfo.classList.remove('hidden');
        btnProcessBatch.disabled = false;
    }

    if (btnRemoveFile) {
        btnRemoveFile.addEventListener('click', (e) => {
            e.stopPropagation();
            archivoSeleccionado = null;
            fileInput.value = '';
            selectedFileInfo.classList.add('hidden');
            btnProcessBatch.disabled = true;
        });
    }

    if (btnProcessBatch) {
        btnProcessBatch.addEventListener('click', async () => {
            if (!archivoSeleccionado) return;

            btnProcessBatch.disabled = true;
            batchSpinner.classList.remove('hidden');

            const formData = new FormData();
            formData.append('archivo', archivoSeleccionado);

            try {
                const res = await fetch('/predecir_lote', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();

                if (res.ok) {
                    mostrarResultadosLote(data);
                } else {
                    alert('Error en diagnóstico por lote: ' + (data.error || 'Desconocido'));
                }
            } catch (err) {
                console.error("Error en lote:", err);
                alert('Error al conectar con el servidor para procesar el lote.');
            } finally {
                btnProcessBatch.disabled = false;
                batchSpinner.classList.add('hidden');
            }
        });
    }

    function mostrarResultadosLote(data) {
        batchResults.classList.remove('hidden');

        // Tarjetas de Resumen KPI
        document.getElementById('rep-total').textContent = data.total_alumnos;
        document.getElementById('rep-alto').textContent = data.total_alto;
        document.getElementById('rep-bajo').textContent = data.total_bajo;
        document.getElementById('rep-pct').textContent = `${data.pct_alto}%`;

        // Tabla de Grupos
        const tbodyGrupos = document.getElementById('tbody-grupos');
        tbodyGrupos.innerHTML = '';
        data.grupos.forEach(g => {
            const tr = document.createElement('tr');
            const esCritico = g.pct_riesgo > 35;
            tr.innerHTML = `
                <td><strong>${g.grado}°</strong></td>
                <td>${g.grupo}</td>
                <td>${g.especialidad.toUpperCase()}</td>
                <td>${g.total}</td>
                <td><strong style="color: var(--alto-color)">${g.en_riesgo}</strong></td>
                <td>${g.pct_riesgo}%</td>
                <td>
                    <span class="${esCritico ? 'badge-alerta-critica' : 'badge-alerta-estable'}">
                        ${esCritico ? '⚠️ ALERTA ELEVADA' : '✓ ESTABLE'}
                    </span>
                </td>
            `;
            tbodyGrupos.appendChild(tr);
        });

        // Tabla Detalle de Estudiantes
        const tbodyDetalle = document.getElementById('tbody-detalle');
        tbodyDetalle.innerHTML = '';
        data.detalle.forEach(a => {
            const tr = document.createElement('tr');
            const esAlto = a.riesgo_predicho === 'ALTO';
            tr.innerHTML = `
                <td>${a.grado}°</td>
                <td>${a.grupo}</td>
                <td>${a.especialidad}</td>
                <td>${a.horas_semana_totales}h</td>
                <td>${a.asistencia_semanal}%</td>
                <td>${a.promedio}</td>
                <td>${a.materias_reprobadas}</td>
                <td>
                    <span class="${esAlto ? 'badge-riesgo-alto' : 'badge-riesgo-bajo'}">
                        ${a.riesgo_predicho}
                    </span>
                </td>
                <td>${a.prob_alto_pct}%</td>
            `;
            tbodyDetalle.appendChild(tr);
        });

        batchResults.scrollIntoView({ behavior: 'smooth' });
    }

    // =========================================================================
    // 6. MODAL DE MÉTRICAS (CURVAS DE APRENDIZAJE)
    // =========================================================================
    if (btnGraficas && graficaModal && closeModal) {
        btnGraficas.addEventListener('click', () => {
            graficaModal.classList.remove('hidden');
        });

        closeModal.addEventListener('click', () => {
            graficaModal.classList.add('hidden');
        });

        window.addEventListener('click', (e) => {
            if (e.target === graficaModal) {
                graficaModal.classList.add('hidden');
            }
        });
    }

});
