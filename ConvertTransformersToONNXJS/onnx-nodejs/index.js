// Setup onnxruntime
// ort = require('onnxruntime-web');
ort = require('onnxruntime-node');

//
const InferenceSession = ort.InferenceSession;
const Tensor = ort.Tensor;

// 
async function main() {
    try {
        // Create session and load the model.
        const session = await InferenceSession.create('../onnx/model.onnx');

        const input = new Array("hello there!");
        inputTensor = new Tensor('string', input, [1]);
        mask = new Tensor('float32', [0], [1]);
        token_type_ids = new Tensor('int32', [0], [1]);
        const feed = { input_ids: inputTensor, attention_mask: mask, token_type_ids: token_type_ids };
        const results = await session.run(feed);
        console.log(results);

        // Run options.
        // const option = createRunOptions();
    } catch (e) {
        console.error(`failed to inference ONNX model: ${e}.`);
    }
}

function createRunOptions() {
    // run options: please refer to the other example for details usage for run options

    // specify log verbose to this inference run
    return { logSeverityLevel: 0 };
}

main();

/*
ort.env.wasm.numThreads = 3;
ort.env.wasm.simd = true;

const options = {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all'
};

const model = '../onnx/model.onnx';

const session = ort.InferenceSession.create(model, options);
session.then(t => {
    downloadingModel = false;
    // warmup the virtual machine
    for (var i=0; i < 10; i++) {

    }
});
*/