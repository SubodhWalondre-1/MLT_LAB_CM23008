let model;
let isDrawing = false;
let canvas, ctx;
let testData;

/* =======================
   CANVAS INITIALIZATION
======================= */
function initCanvas() {
    canvas = document.getElementById('drawCanvas');
    ctx = canvas.getContext('2d');

    resetCanvasStyle();

    const start = (e) => {
        isDrawing = true;
        const { x, y } = getPos(e);
        ctx.beginPath();
        ctx.moveTo(x, y);
    };

    const move = (e) => {
        if (!isDrawing) return;
        const { x, y } = getPos(e);
        ctx.lineTo(x, y);
        ctx.stroke();
    };

    const stop = () => (isDrawing = false);

    canvas.addEventListener('mousedown', start);
    canvas.addEventListener('mousemove', move);
    canvas.addEventListener('mouseup', stop);
    canvas.addEventListener('mouseout', stop);

    canvas.addEventListener('touchstart', (e) => handleTouch(e, start));
    canvas.addEventListener('touchmove', (e) => handleTouch(e, move));
    canvas.addEventListener('touchend', stop);
}

function resetCanvasStyle() {
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 20;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
}

function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: (e.clientX || e.touches[0].clientX) - rect.left,
        y: (e.clientY || e.touches[0].clientY) - rect.top
    };
}

function handleTouch(e, callback) {
    e.preventDefault();
    callback(e);
}

function clearCanvas() {
    resetCanvasStyle();
    document.getElementById('predictionResult').style.display = 'none';
}

/* =======================
   MODEL CREATION
======================= */
function createModel() {
    const model = tf.sequential();

    model.add(tf.layers.conv2d({
        inputShape: [28, 28, 1],
        filters: 32,
        kernelSize: 3,
        activation: 'relu',
        padding: 'same'
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

    model.add(tf.layers.conv2d({
        filters: 64,
        kernelSize: 3,
        activation: 'relu',
        padding: 'same'
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.5 }));
    model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

    return model;
}

/* =======================
   PREDICTION FROM CANVAS
======================= */
async function predictDigit() {
    if (!model) {
        console.warn('Model not trained yet');
        return;
    }

    const tensor = tf.tidy(() => {
        const img = tf.browser
            .fromPixels(ctx.getImageData(0, 0, canvas.width, canvas.height), 1)
            .resizeBilinear([28, 28])
            .div(255)
            .sub(1)
            .abs()
            .reshape([1, 28, 28, 1]);

        return img;
    });

    const prediction = model.predict(tensor);
    const probs = await prediction.data();
    const digit = probs.indexOf(Math.max(...probs));
    const confidence = (probs[digit] * 100).toFixed(2);

    /* 🔥 CONSOLE OUTPUT (REQUESTED) */
    console.group('🧠 Digit Prediction');
    console.log('Predicted Digit:', digit);
    console.log('Confidence:', `${confidence}%`);
    console.table(
        probs.map((p, i) => ({
            Digit: i,
            Probability: (p * 100).toFixed(2) + '%'
        }))
    );
    console.groupEnd();

    /* UI UPDATE */
    document.getElementById('predictionResult').style.display = 'block';
    document.getElementById('predictedDigit').textContent = digit;
    document.getElementById('predictionConfidence').textContent =
        `Confidence: ${confidence}%`;

    tensor.dispose();
    prediction.dispose();
}

/* =======================
   INIT
======================= */
window.onload = () => {
    initCanvas();
    console.log('✅ TensorFlow.js:', tf.version.tfjs);
    console.log('✏️ Draw a digit → Click Predict');
};
