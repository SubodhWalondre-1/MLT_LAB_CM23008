console.log("TensorFlow.js version:", tf.version.tfjs);

function addTensors() {

    // Read input values
    const inputA = document.getElementById("inputA").value;
    const inputB = document.getElementById("inputB").value;

    // Convert input string to number array
    const arrA = inputA.split(" ").map(Number);
    const arrB = inputB.split(" ").map(Number);

    // Create tensors
    const tensorA = tf.tensor(arrA);
    const tensorB = tf.tensor(arrB);

    // Perform addition
    const result = tf.add(tensorA, tensorB);

    // Print result in console
    result.print();

    // Display result on screen
    document.getElementById("output").innerText =
        "Result Tensor: " + result.toString();

    // Free memory
    tensorA.dispose();
    tensorB.dispose();
    result.dispose();
}
