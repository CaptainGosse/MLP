// multi-layer perceptron

const box = document.getElementById('box');
const pred = document.getElementById('prediction'); // Corrected the ID to 'prediction'
const pixel = [];
let isDrawing = false;
let btn = document.getElementsByClassName('btn');

function getPixelArray() {
    const pixelArray = [];
    for (let i = 0; i < 784; i++) {
        pixelArray.push(pixel[`btn${i}`] || 0);
    }
    return pixelArray;
}
async function loadJSON(file) {
    const response = await fetch(file);
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
}

async function predict(x_test) {
    const x = getPixelArray()
    const xTensor = tf.tensor([x], [1, 784]);

    // Load weights and biases
    const w1 = tf.tensor(await loadJSON('w1.json'));
    const b1 = tf.tensor(await loadJSON('b1.json'));
    const w2 = tf.tensor(await loadJSON('w2.json'));
    const b2 = tf.tensor(await loadJSON('b2.json'));
    const w3 = tf.tensor(await loadJSON('w3.json'));
    const b3 = tf.tensor(await loadJSON('b3.json'));

    // Activation functions
    function relu(x) {
        return x.relu();
    }

    function softmax(x) {
        return x.softmax();
    }

    const start = performance.now();

    const z1 = xTensor.matMul(w1).add(b1);
    const a1 = relu(z1);

    // Second layer
    const z2 = a1.matMul(w2).add(b2);
    const a2 = relu(z2);

    // Third layer
    const z3 = a2.matMul(w3).add(b3);
    const y_pred_test = softmax(z3);

    console.log(`Time taken: ${(performance.now() - start) / 1000} seconds`);
    return y_pred_test.argMax(1).dataSync()[0];
}

function make_grid() {
    for (let i = 0; i < 784; i++) {
        const btn = document.createElement('button');
        btn.className = 'btn';
        btn.id = `btn${i}`;
        box.appendChild(btn);
        pixel[btn.id] = 0; // Initialize all buttons as unpainted

        btn.addEventListener('mousedown', (event) => {
            isDrawing = true;
            paint(event.target);
        });

        btn.addEventListener('mouseenter', (event) => {
            if (isDrawing) paint(event.target);
        });
    }
}

// Paint a button and update pixel tracking
function paint(btn) {
    const id = parseInt(btn.id.replace('btn', ''));
    const row = Math.floor(id / 28);
    const col = id % 28;

    // Paint the center button black
    btn.style.backgroundColor = '#F4EDD3';
    pixel[btn.id] = 1;

    // Paint the surrounding buttons gray
    const offsets = [
        [-1, -1], [-1, 0], [-1, 1],
        [0, -1], /*[0, 0],*/ [0, 1],
        [1, -1], [1, 0], [1, 1]
    ];

    offsets.forEach(offset => {
        const newRow = row + offset[0];
        const newCol = col + offset[1];
        if (newRow >= 0 && newRow < 28 && newCol >= 0 && newCol < 28) {
            const newId = newRow * 28 + newCol;
            const newBtn = document.getElementById(`btn${newId}`);
            if (newBtn) {
                newBtn.style.backgroundColor = 'gray';
                pixel[newBtn.id] = Math.floor(Math.random() * (7 - 2 + 1)) + 2;
            }
        }
    });
}

// Stop drawing on mouseup
document.addEventListener('mouseup', () => {
    isDrawing = false;
    console.log(pixel);
    
    predict(pixel).then(prediction => {
        pred.textContent = `MLP think the number is: ${prediction}`;
    });
});

// Clear the grid
function clear_grid() {
    const grid = document.querySelectorAll('.btn');
    grid.forEach(grid => {
        grid.style.backgroundColor = '#A5BFCC';
        pixel[grid.id] = 0; // Reset pixel tracking
    });
}

document.getElementById('predict-btn').addEventListener('click', () => {
    predict(pixel).then(prediction => {
        pred.textContent = `MLP think the number is: ${prediction}`;
    });
});

document.getElementById('clear-btn').addEventListener('click', clear_grid);

// Initialize the grid on page load
make_grid();



