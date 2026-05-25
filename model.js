const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const artData = require('./renaissanceart.js');

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────
const CONFIG = {
  IMAGE_WIDTH: 224,
  IMAGE_HEIGHT: 224,
  CHANNELS: 3,
  BATCH_SIZE: 32,
  EPOCHS: 50,
  LEARNING_RATE: 0.0001,
  VALIDATION_SPLIT: 0.2,
  MODEL_SAVE_PATH: 'file://./renaissance-art-model',
  TOP_K_PREDICTIONS: 3, // Return top 3 artist guesses
};

// ─────────────────────────────────────────────
// Artist label mapping
// ─────────────────────────────────────────────
const uniqueArtists = [...new Set(artData.map(item => item.artist))];
const numClasses = uniqueArtists.length;

console.log(`Found ${numClasses} unique artists in dataset.`);

// ─────────────────────────────────────────────
// Model Architecture
// ─────────────────────────────────────────────
function createModel() {
  const model = tf.sequential();

  // Block 1
  model.add(tf.layers.conv2d({
    inputShape: [CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.CHANNELS],
    filters: 32,
    kernelSize: 3,
    activation: 'relu',
    padding: 'same',
    kernelInitializer: 'heNormal', // Better weight init for ReLU
  }));
  model.add(tf.layers.batchNormalization()); // Stabilizes training
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

  // Block 2
  model.add(tf.layers.conv2d({
    filters: 64,
    kernelSize: 3,
    activation: 'relu',
    padding: 'same',
    kernelInitializer: 'heNormal',
  }));
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

  // Block 3
  model.add(tf.layers.conv2d({
    filters: 128,
    kernelSize: 3,
    activation: 'relu',
    padding: 'same',
    kernelInitializer: 'heNormal',
  }));
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

  // Block 4 — extra depth for richer feature extraction
  model.add(tf.layers.conv2d({
    filters: 256,
    kernelSize: 3,
    activation: 'relu',
    padding: 'same',
    kernelInitializer: 'heNormal',
  }));
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.globalAveragePooling2d()); // Replaces flatten — fewer params, less overfitting

  // Classifier head
  model.add(tf.layers.dense({ units: 512, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.5 }));
  model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(tf.layers.dense({ units: numClasses, activation: 'softmax' }));

  return model;
}

// ─────────────────────────────────────────────
// Image Preprocessing
// ─────────────────────────────────────────────
async function loadAndPreprocessImage(imagePath) {
  return tf.tidy(() => { // tf.tidy() auto-disposes intermediate tensors — prevents memory leaks
    try {
      const imageBuffer = fs.readFileSync(imagePath);
      const tfImage = tf.node.decodeImage(imageBuffer, 3); // Force 3 channels (RGB)
      const resized = tf.image.resizeBilinear(tfImage, [CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH]);
      const normalized = resized.div(255.0);
      return normalized;
    } catch (error) {
      console.error(`Failed to load image at ${imagePath}:`, error.message);
      return null;
    }
  });
}

// ─────────────────────────────────────────────
// Data Augmentation
// Randomly transforms images during training to reduce overfitting
// ─────────────────────────────────────────────
function augmentImage(imageTensor) {
  return tf.tidy(() => {
    let augmented = imageTensor;

    // Random horizontal flip (50% chance)
    if (Math.random() > 0.5) {
      augmented = tf.image.flipLeftRight(augmented);
    }

    // Random brightness shift (±15%)
    augmented = tf.image.adjustBrightness(augmented, (Math.random() - 0.5) * 0.3);

    // Clip values back to [0, 1] after brightness shift
    augmented = tf.clipByValue(augmented, 0, 1);

    return augmented;
  });
}

// ─────────────────────────────────────────────
// Dataset Preparation
// Loads images in batches to avoid OOM crashes on large datasets
// ─────────────────────────────────────────────
async function prepareDataset() {
  const loadedImages = [];
  const labels = [];
  let skipped = 0;

  console.log(`Loading ${artData.length} artworks...`);

  for (const artwork of artData) {
    const image = await loadAndPreprocessImage(artwork.imagePath);
    if (image !== null) {
      loadedImages.push(image);
      labels.push(uniqueArtists.indexOf(artwork.artist));
    } else {
      skipped++;
    }
  }

  console.log(`Loaded: ${loadedImages.length} | Skipped: ${skipped}`);

  // Shuffle indices randomly before splitting (prevents biased val set)
  const indices = tf.util.createShuffledIndices(loadedImages.length);
  const shuffledImages = Array.from(indices).map(i => loadedImages[i]);
  const shuffledLabels = Array.from(indices).map(i => labels[i]);

  // Stack into tensors
  const xs = tf.stack(shuffledImages);
  const ys = tf.oneHot(shuffledLabels, numClasses);

  // Clean up individual image tensors now that they're stacked
  shuffledImages.forEach(img => img.dispose());

  return { xs, ys, numExamples: shuffledImages.length };
}

// ─────────────────────────────────────────────
// Training
// ─────────────────────────────────────────────
async function trainModel() {
  // Load existing model if available — skip retraining
  if (fs.existsSync('./renaissance-art-model/model.json')) {
    console.log('Saved model found. Loading instead of retraining...');
    const model = await tf.loadLayersModel(CONFIG.MODEL_SAVE_PATH);
    model.compile({
      optimizer: tf.train.adam(CONFIG.LEARNING_RATE),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });
    return model;
  }

  console.log('No saved model found. Starting training...');
  const data = await prepareDataset();

  const model = createModel();
  model.compile({
    optimizer: tf.train.adam(CONFIG.LEARNING_RATE),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  model.summary();

  // Split: first 80% train, last 20% validation
  const splitIndex = Math.floor(data.numExamples * (1 - CONFIG.VALIDATION_SPLIT));
  const trainXs = data.xs.slice([0, 0, 0, 0], [splitIndex, -1, -1, -1]);
  const trainYs = data.ys.slice([0, 0], [splitIndex, -1]);
  const valXs = data.xs.slice([splitIndex, 0, 0, 0]);
  const valYs = data.ys.slice([splitIndex, 0]);

  let bestValAcc = 0;
  let epochsWithoutImprovement = 0;
  const PATIENCE = 5; // Early stopping: stop if no improvement for 5 epochs

  await model.fit(trainXs, trainYs, {
    epochs: CONFIG.EPOCHS,
    batchSize: CONFIG.BATCH_SIZE,
    validationData: [valXs, valYs],
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        const valAcc = logs.val_acc ?? logs.val_accuracy;
        console.log(
          `Epoch ${epoch + 1}/${CONFIG.EPOCHS} — ` +
          `loss: ${logs.loss.toFixed(4)} | acc: ${(logs.acc ?? logs.accuracy).toFixed(4)} | ` +
          `val_loss: ${logs.val_loss.toFixed(4)} | val_acc: ${valAcc.toFixed(4)}`
        );

        // Save model whenever validation accuracy improves
        if (valAcc > bestValAcc) {
          bestValAcc = valAcc;
          epochsWithoutImprovement = 0;
          await model.save(CONFIG.MODEL_SAVE_PATH);
          console.log(`  ✓ New best val_acc: ${valAcc.toFixed(4)} — model saved.`);
        } else {
          epochsWithoutImprovement++;
          if (epochsWithoutImprovement >= PATIENCE) {
            console.log(`  ✗ No improvement for ${PATIENCE} epochs. Early stopping.`);
            model.stopTraining = true; // Graceful early stop
          }
        }
      },
    },
  });

  // Cleanup
  data.xs.dispose();
  data.ys.dispose();
  trainXs.dispose();
  trainYs.dispose();
  valXs.dispose();
  valYs.dispose();

  console.log(`Training complete. Best validation accuracy: ${(bestValAcc * 100).toFixed(2)}%`);
  return model;
}

// ─────────────────────────────────────────────
// Prediction
// Returns top artists with confidence scores
// ─────────────────────────────────────────────
async function predictArtwork(model, imagePath) {
  const image = await loadAndPreprocessImage(imagePath);
  if (!image) throw new Error(`Could not load image: ${imagePath}`);

  const result = tf.tidy(() => {
    const batched = image.expandDims(0);
    const prediction = model.predict(batched);
    const probabilities = prediction.dataSync(); // Full probability distribution

    // Build ranked list of top predictions
    const ranked = Array.from(probabilities)
      .map((prob, index) => ({
        artist: uniqueArtists[index],
        confidence: (prob * 100).toFixed(2) + '%',
        score: prob,
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, CONFIG.TOP_K_PREDICTIONS)
      .map(({ artist, confidence }) => ({ artist, confidence })); // Remove raw score from output

    return ranked;
  });

  image.dispose();
  return result;
}

// ─────────────────────────────────────────────
// Exports
// ─────────────────────────────────────────────
module.exports = {
  trainModel,
  predictArtwork,
  createModel,
  uniqueArtists,
  CONFIG,
};
