import {MnistData} from './data.js';

async function showExamples(data) {
  // Creaamos un contenedor para la visualizacion
  const surface =
    tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data'});  

  // Obtenemos los ejemplos
  const examples = data.nextTestBatch(20);
  const numExamples = examples.xs.shape[0];
  
  // Creamos un lienzo para renderizar los ejemplos
  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      // Formateamos la imagen a un tamaño en pixel de 28x28
      return examples.xs
        .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([28, 28, 1]);
    });
    
    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = 'margin: 4px;';
    await tf.browser.toPixels(imageTensor, canvas);
    surface.drawArea.appendChild(canvas);

    imageTensor.dispose();
  }
}

async function run() {  
  const data = new MnistData();
  await data.load();
  await showExamples(data);

  const model = getModel();

  await showAccuracy(model, data);
await showConfusion(model, data);

tfvis.show.modelSummary({name: 'Model Architecture', tab: 'Model'}, model);
  
await train(model, data);
}

function getModel() {
    const model = tf.sequential();
    
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const IMAGE_CHANNELS = 1;  
    
    // En la primera capa de nuestra red neuronal convulcional 
    // tenemos que especificar como llegaran los daots, luego debemos especificar los parametros 
    // que la operacion de la capa utilizara o procesara.
    model.add(tf.layers.conv2d({
      inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
      kernelSize: 5,
      filters: 8,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));
  
    // La capa de Limite Maximo actua como un reductor para los datos usando valores topes 
    // en una region en lugar de normalizar.  
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    
    // Repetimos otra capa de convulsion y limite maximo. 
    // PD: tenemos mas metodos de filtrado en la capa de convulsion.
    model.add(tf.layers.conv2d({
      kernelSize: 5,
      filters: 16,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    
    // Ahora normalizamos los datos de salida desde el filtro 2D hacia un vector 1D
    // Para poder prepararlos para la insercion en nuestra ultima capa, esta es una practica comun para cuando se esta insertando
    // una gran cantidad de datos a la capa final de clasificacion.
    model.add(tf.layers.flatten());
  
    // Nuestra ultima capa contiene una capa densa de 9 unidades, una por cada clase de dato
    // Clase exterior (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
    //Los datos del modelo seran representados de manera numerica -
    const NUM_OUTPUT_CLASSES = 8;
    model.add(tf.layers.dense({
      units: NUM_OUTPUT_CLASSES,
      kernelInitializer: 'varianceScaling',
      activation: 'softmax'
    }));
  
    
    // Seleccionaremos un optimizados para medir la perdida y la precision de nuestra red.
    // Luego compilaremos y retornaremos el modelo.
    const optimizer = tf.train.adam();
    model.compile({
      optimizer: optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });
  
    return model;
  }

  async function train(model, data) {
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const container = {
      name: 'Model Training', tab: 'Model', styles: { height: '1000px' }
    };
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
    
    const BATCH_SIZE = 512;
    const TRAIN_DATA_SIZE = 5500;
    const TEST_DATA_SIZE = 1000;
  
    const [trainXs, trainYs] = tf.tidy(() => {
      const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
      return [
        d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
        d.labels
      ];
    });
  
    const [testXs, testYs] = tf.tidy(() => {
      const d = data.nextTestBatch(TEST_DATA_SIZE);
      return [
        d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
        d.labels
      ];
    });
  
    return model.fit(trainXs, trainYs, {
      batchSize: BATCH_SIZE,
      validationData: [testXs, testYs],
      epochs: 10,
      shuffle: true,
      callbacks: fitCallbacks
    });
  }

  const classNames = ['Buy and sell Mobiles', 'Fix mobiles', 'Buy and sell accesories for mobiles', 'Fix accesories for mobiles', 
  'Buy and sell PCs', 'Fix PCs', 'Buy and sell accesories for PCs', 'Fix accesories for PCs'];

function doPrediction(model, data, testDataSize = 1000) {
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const testData = data.nextTestBatch(testDataSize);
  const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
  const labels = testData.labels.argMax(-1);
  const preds = model.predict(testxs).argMax(-1);

  testxs.dispose();
  return [preds, labels];
}


async function showAccuracy(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
  const container = {name: 'Accuracy', tab: 'Evaluation'};
  tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

  labels.dispose();
}

async function showConfusion(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
  const container = {name: 'Confusion Matrix', tab: 'Evaluation'};
  tfvis.render.confusionMatrix(container, {values: confusionMatrix, tickLabels: classNames});

  labels.dispose();
}


document.addEventListener('DOMContentLoaded', run);