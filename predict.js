const tf = require('@tensorflow/tfjs-node');

const CLASSES = ["team03", "team18", "team21", "team36"];
const epochs = 100;
const labels = [1, 2, 3, 4];

async function loadData() {
    const dataset = tf.data.csv('file://./data.csv', { hasHeader: true });
    const v = await dataset.take(-1).toArray();

    let tempx = []
    let tempy = []

    function createOnehot(label) {
        let onehot = []
        for (let j = 0; j < 4; j++) {
            if (label == labels[j]) {
                onehot.push(1);
            } else {
                onehot.push(0);
            }
        }
        return onehot;
    }

    v.forEach((line) => {
        console.log(line);
        temp0 = [line['rssi1'], line['rssi2'], line['rssi3'], line['rssi4']];

        temp1 = createOnehot(line['index_label']);
        tempx.push(temp0);
        tempy.push(temp1);
    });
    console.log(tempx);
    console.log(tempy);
    let xs = tf.tensor2d(tempx);
    let ys = tf.tensor2d(tempy);
    /*
    let xs = tf.randomNormal([10, 4]);
    let ys = tf.randomNormal([10, 4]);
    */
    return [xs, ys]
}

function createModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 4, activation: 'linear', inputShape: [4] }));
    model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 32, activation: 'linear' }));
    model.add(tf.layers.dense({ units: 4, activation: 'softmax' }));

    const optimizer = tf.train.adam(0.01);
    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });
    return model
}

async function trainModel(model, xs, ys, epochs) {
    const history = await model.fit(xs, ys, {
        epochs: epochs,
        callbacks: {
            onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}`)
        }
    });

    return history['history']['loss']
}

async function predict(model, inputData) {
    const predictOut = model.predict(inputData);
    const logits = Array.from(predictOut.dataSync());

    const winner = CLASSES[predictOut.argMax(-1).dataSync()[0]];
    return winner;
}

module.exports = { loadData, createModel, trainModel };

(async() => {

    //2.1 การสร้าง model ด้วย TensorFlow.js
    let model = createModel()

    //2.2 สร้าง classification ด้วย Node.js เพื่อดึงข้อมูล CSV มา train

    let [xs, ys] = await loadData()
    xs.print()
    ys.print()

    const loss = await trainModel(model, xs, ys, epochs);

    //2.3 บันทึกไฟล์ model
    const saveResult = await model.save('file://modelW');

    //3.1 การจำแนกตำแหน่ง
    let xv = tf.randomNormal([1, 4]);
    var yv = await predict(model, xv)
    console.log('Test ')
    xv.print()
    console.log('Predict ')
    console.log(yv);

    //3.2 สร้างหน้าเว็บ HTML ด้วย Node.js

    //3.3 ใช้ไลบรารี TensorFlow.js เพื่อ predict ตำแหน่ง model จากข้อ 2
    //const model = await tf.loadLayersModel('localstorage://modelW');

    //3.4 มีความแม่นยำในการจำแนก 2 ตำแหน่ง (สุ่มถาม)

})();