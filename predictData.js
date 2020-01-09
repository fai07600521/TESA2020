
const tf = require('@tensorflow/tfjs-node');
const model =  tf.loadLayersModel('file://./tfjs_model/model.json');
async function predict(model, inputData) {
    const predictOut = model.predict(inputData);
    const logits = Array.from(predictOut.dataSync());

    const winner = CLASSES[predictOut.argMax(-1).dataSync()[0]];
    return winner;
}

console.log(model);
