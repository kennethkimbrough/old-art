# ═══════════════════════════════════════════════════════════════════════════════
# Renart - Renaissance Art Classifier
# main.r — Complete training pipeline
#
# Datasets:
#   - WikiArt via Hugging Face (huggan/wikiart)
#   - National Gallery of Art (NGA) open data + IIIF image API
#
# One-time setup (run these in your terminal before running this script):
#   pip install datasets huggingface_hub pillow requests pandas tqdm
#   Rscript -e "install.packages(c('keras','magrittr','jsonlite','ggplot2','caret','scales'))"
#
# Run:
#   Rscript main.r
#   Rscript main.r predict path/to/painting.jpg   (skip training, predict only)
# ═══════════════════════════════════════════════════════════════════════════════

suppressPackageStartupMessages({
  library(keras)
  library(magrittr)
  library(jsonlite)
  library(ggplot2)
  library(scales)
})

# ───────────────────────────────────────────────────────────────────────────────
# 0. Parse command-line arguments
# ───────────────────────────────────────────────────────────────────────────────
args <- commandArgs(trailingOnly = TRUE)
PREDICT_ONLY  <- "--predict" %in% args
PREDICT_IMAGE <- if (PREDICT_ONLY) args[which(args == "--predict") + 1] else NULL

# ───────────────────────────────────────────────────────────────────────────────
# 1. Configuration
#    All settings in one place
# ───────────────────────────────────────────────────────────────────────────────
CONFIG <- list(

  # Paths
  data_dir         = "./data/renaissance",
  train_dir        = "./data/renaissance/train",
  val_dir          = "./data/renaissance/val",
  model_save_path  = "./models/renart_best_model.h5",
  final_model_path = "./models/renart_final_model.h5",
  results_dir      = "./results",
  dataset_script   = "./renart_dataset.py",   # The Python dual-source downloader
  stats_file       = "./data/renaissance/dataset_stats.json",
  class_index_file = "./results/class_indices.json",

  # Image settings — must match model.js
  image_size = c(224, 224),
  channels   = 3,

  # Training hyperparameters
  batch_size    = 32,
  epochs        = 50,
  learning_rate = 0.0001,
  patience      = 7,     # Early stopping: stop after N epochs without improvement
  lr_patience   = 3,     # Reduce LR after N epochs without improvement
  lr_factor     = 0.5,   # Multiply LR by this when reducing
  min_lr        = 1e-7,

  # Target Renaissance artists — must match renart_dataset.py
  target_artists = c(
    "Leonardo Da Vinci",
    "Michelangelo",
    "Raphael",
    "Titian",
    "Sandro Botticelli",
    "Caravaggio",
    "Tintoretto",
    "Paolo Veronese",
    "Giorgione",
    "Fra Angelico",
    "Masaccio",
    "Andrea Mantegna",
    "Giovanni Bellini",
    "Filippino Lippi",
    "Luca Signorelli",
    "Pietro Perugino",
    "Domenico Ghirlandaio",
    "Jacopo Pontormo",
    "Piero Della Francesca",
    "Lorenzo Lotto"
  )
)

# ───────────────────────────────────────────────────────────────────────────────
# 2. Dataset Download
#    Calls renart_dataset.py which pulls from WikiArt + NGA.
#    Skips automatically if dataset_stats.json already exists.
# ───────────────────────────────────────────────────────────────────────────────
download_dataset <- function(config) {

  if (file.exists(config$stats_file)) {
    stats <- fromJSON(config$stats_file)
    cat(sprintf(
      "Dataset already downloaded: %d images across %d artists.\n",
      stats$total_images, stats$num_artists
    ))
    cat("To re-download, delete:", config$stats_file, "\n\n")
    return(invisible(TRUE))
  }

  cat("Dataset not found. Running dual-source download (WikiArt + NGA)...\n")
  cat("This will take 10–30 minutes on first run.\n\n")

  # Check the Python downloader script exists
  if (!file.exists(config$dataset_script)) {
    stop(paste(
      "renart_dataset.py not found at:", config$dataset_script,
      "\nMake sure renart_dataset.py is in the same folder as main.r"
    ))
  }

  # Check Python dependencies
  dep_check <- system("python3 -c 'import datasets, PIL, requests, pandas, tqdm'",
                      ignore.stdout = TRUE, ignore.stderr = TRUE)
  if (dep_check != 0) {
    stop(paste(
      "Missing Python dependencies. Run:\n",
      "  pip install datasets huggingface_hub pillow requests pandas tqdm"
    ))
  }

  exit_code <- system(paste("python3", config$dataset_script))

  if (exit_code != 0 || !file.exists(config$stats_file)) {
    stop("Dataset download failed. Check the error messages above.")
  }

  stats <- fromJSON(config$stats_file)
  cat(sprintf(
    "\nDownload complete: %d images, %d artists.\n\n",
    stats$total_images, stats$num_artists
  ))

  # Print per-artist summary
  cat("Per-artist image counts:\n")
  counts <- stats$counts
  for (artist in names(counts)) {
    bar   <- paste(rep("█", counts[[artist]] %/% 15), collapse = "")
    cat(sprintf("  %-32s %3d  %s\n", artist, counts[[artist]], bar))
  }
  cat("\n")

  invisible(TRUE)
}

# ───────────────────────────────────────────────────────────────────────────────
# 3. Data Generators
#    Train: heavy augmentation to fight overfitting on a relatively small dataset
#    Val:   no augmentation — evaluate on real unmodified images
# ───────────────────────────────────────────────────────────────────────────────
create_generators <- function(config) {

  cat("Creating data generators...\n")

  train_datagen <- image_data_generator(
    rescale            = 1/255,
    rotation_range     = 20,          # Rotate up to ±20 degrees
    width_shift_range  = 0.15,        # Shift horizontally up to 15%
    height_shift_range = 0.15,        # Shift vertically up to 15%
    horizontal_flip    = TRUE,        # Mirror — valid transformation for paintings
    brightness_range   = c(0.75, 1.25), # Vary exposure ±25%
    zoom_range         = 0.15,        # Zoom in/out up to 15%
    shear_range        = 0.1,         # Slight shear distortion
    channel_shift_range = 20,         # Slight colour shift — handles photo variations
    fill_mode          = "reflect"    # Mirror-fill empty areas after rotation/shift
  )

  val_datagen <- image_data_generator(rescale = 1/255)

  train_generator <- flow_images_from_directory(
    directory   = config$train_dir,
    generator   = train_datagen,
    target_size = config$image_size,
    batch_size  = config$batch_size,
    class_mode  = "categorical",
    shuffle     = TRUE,
    seed        = 42
  )

  val_generator <- flow_images_from_directory(
    directory   = config$val_dir,
    generator   = val_datagen,
    target_size = config$image_size,
    batch_size  = config$batch_size,
    class_mode  = "categorical",
    shuffle     = FALSE    # Keep order consistent for evaluation
  )

  num_classes <- length(train_generator$class_indices)
  cat(sprintf("  Train images: %d\n", train_generator$n))
  cat(sprintf("  Val images:   %d\n", val_generator$n))
  cat(sprintf("  Classes:      %d artists\n\n", num_classes))

  list(train = train_generator, val = val_generator, num_classes = num_classes)
}

# ───────────────────────────────────────────────────────────────────────────────
# 4. Model Architecture
#    4-block CNN with BatchNormalization after every conv layer.
#    GlobalAveragePooling instead of Flatten — fewer parameters, less overfitting.
#    Matches the architecture in model.js so R and JS models are consistent.
# ───────────────────────────────────────────────────────────────────────────────
build_model <- function(num_classes, config) {

  cat("Building model architecture...\n")

  input_shape <- c(config$image_size, config$channels)  # c(224, 224, 3)

  model <- keras_model_sequential(name = "renart_cnn") %>%

    # ── Block 1: low-level features (edges, colours, textures) ──
    layer_conv_2d(
      filters = 32, kernel_size = c(3, 3), padding = "same",
      activation = "relu", input_shape = input_shape,
      kernel_initializer = "he_normal", name = "conv1"
    ) %>%
    layer_batch_normalization(name = "bn1") %>%
    layer_max_pooling_2d(pool_size = c(2, 2), name = "pool1") %>%

    # ── Block 2: mid-level features (brush strokes, shapes) ──
    layer_conv_2d(
      filters = 64, kernel_size = c(3, 3), padding = "same",
      activation = "relu", kernel_initializer = "he_normal", name = "conv2"
    ) %>%
    layer_batch_normalization(name = "bn2") %>%
    layer_max_pooling_2d(pool_size = c(2, 2), name = "pool2") %>%

    # ── Block 3: high-level features (composition, style patterns) ──
    layer_conv_2d(
      filters = 128, kernel_size = c(3, 3), padding = "same",
      activation = "relu", kernel_initializer = "he_normal", name = "conv3"
    ) %>%
    layer_batch_normalization(name = "bn3") %>%
    layer_max_pooling_2d(pool_size = c(2, 2), name = "pool3") %>%

    # ── Block 4: abstract features (artist-specific style signatures) ──
    layer_conv_2d(
      filters = 256, kernel_size = c(3, 3), padding = "same",
      activation = "relu", kernel_initializer = "he_normal", name = "conv4"
    ) %>%
    layer_batch_normalization(name = "bn4") %>%
    layer_global_average_pooling_2d(name = "gap") %>%

    # ── Classifier head ──
    layer_dense(units = 512, activation = "relu",    name = "dense1") %>%
    layer_dropout(rate = 0.5,                         name = "drop1") %>%
    layer_dense(units = 256, activation = "relu",    name = "dense2") %>%
    layer_dropout(rate = 0.3,                         name = "drop2") %>%
    layer_dense(units = num_classes, activation = "softmax", name = "output")

  model %>% compile(
    optimizer = optimizer_adam(learning_rate = config$learning_rate),
    loss      = "categorical_crossentropy",
    metrics   = c("accuracy")
  )

  summary(model)
  cat("\n")

  model
}

# ───────────────────────────────────────────────────────────────────────────────
# 5. Training
#    Three callbacks:
#      - EarlyStopping:      stops when val_accuracy plateaus
#      - ModelCheckpoint:    saves the single best model to disk
#      - ReduceLROnPlateau:  halves learning rate when val_loss stalls
# ───────────────────────────────────────────────────────────────────────────────
train_model <- function(model, generators, config) {

  dir.create(dirname(config$model_save_path),  recursive = TRUE, showWarnings = FALSE)
  dir.create(config$results_dir,               recursive = TRUE, showWarnings = FALSE)

  callbacks <- list(

    callback_early_stopping(
      monitor              = "val_accuracy",
      patience             = config$patience,
      restore_best_weights = TRUE,   # Auto-rollback to best epoch
      verbose              = 1
    ),

    callback_model_checkpoint(
      filepath       = config$model_save_path,
      monitor        = "val_accuracy",
      save_best_only = TRUE,
      verbose        = 1
    ),

    callback_reduce_lr_on_plateau(
      monitor  = "val_loss",
      factor   = config$lr_factor,
      patience = config$lr_patience,
      min_lr   = config$min_lr,
      verbose  = 1
    ),

    # Log each epoch to a CSV for later analysis
    callback_csv_logger(
      filename = file.path(config$results_dir, "training_log.csv"),
      append   = FALSE
    )
  )

  cat("Starting training...\n")
  cat(sprintf("  Epochs:        up to %d (early stopping patience: %d)\n",
              config$epochs, config$patience))
  cat(sprintf("  Batch size:    %d\n", config$batch_size))
  cat(sprintf("  Learning rate: %g\n\n", config$learning_rate))

  history <- model %>% fit(
    generators$train,
    epochs           = config$epochs,
    validation_data  = generators$val,
    steps_per_epoch  = ceiling(generators$train$n / config$batch_size),
    validation_steps = ceiling(generators$val$n   / config$batch_size),
    callbacks        = callbacks,
    verbose          = 1
  )

  history
}

# ───────────────────────────────────────────────────────────────────────────────
# 6. Evaluation
#    - Final accuracy + loss on validation set
#    - Per-class accuracy table
#    - Confusion matrix heatmap (saved as PNG)
#    - Training history curves (saved as PNG)
# ───────────────────────────────────────────────────────────────────────────────
evaluate_model <- function(model, generators, history, config) {

  cat("\n── Evaluation ────────────────────────────────\n")

  # Overall accuracy
  results <- model %>% evaluate(generators$val,
                                steps   = ceiling(generators$val$n / config$batch_size),
                                verbose = 0)
  cat(sprintf("  Val Loss:     %.4f\n", results["loss"]))
  cat(sprintf("  Val Accuracy: %.2f%%\n\n", results["accuracy"] * 100))

  # ── Per-class accuracy ──
  cat("Per-artist accuracy:\n")
  class_names <- names(generators$val$class_indices)

  # Generate predictions on the entire val set
  steps    <- ceiling(generators$val$n / config$batch_size)
  preds_raw <- model %>% predict(generators$val, steps = steps, verbose = 0)
  pred_classes  <- apply(preds_raw, 1, which.max) - 1L  # 0-indexed

  # Rebuild true labels from generator (reset so order is consistent)
  generators$val$reset()
  true_labels <- c()
  for (i in seq_len(steps)) {
    batch       <- generator_next(generators$val)
    batch_labels <- apply(batch[[2]], 1, which.max) - 1L
    true_labels  <- c(true_labels, batch_labels)
  }

  # Trim to same length (last batch may be partial)
  n <- min(length(pred_classes), length(true_labels))
  pred_classes <- pred_classes[1:n]
  true_labels  <- true_labels[1:n]

  # Per-class accuracy
  per_class_acc <- sapply(seq_along(class_names) - 1, function(cls) {
    mask <- true_labels == cls
    if (sum(mask) == 0) return(NA)
    mean(pred_classes[mask] == cls)
  })

  acc_df <- data.frame(
    Artist   = class_names,
    Accuracy = round(per_class_acc * 100, 1),
    Count    = sapply(seq_along(class_names) - 1, function(cls) sum(true_labels == cls))
  )
  acc_df <- acc_df[order(-acc_df$Accuracy), ]

  for (i in seq_len(nrow(acc_df))) {
    bar <- paste(rep("█", floor(acc_df$Accuracy[i] / 5)), collapse = "")
    cat(sprintf("  %-32s %5.1f%%  %s\n",
                acc_df$Artist[i], acc_df$Accuracy[i], bar))
  }

  # ── Training history plot ──
  history_df        <- as.data.frame(history$metrics)
  history_df$epoch  <- seq_len(nrow(history_df))

  # Reshape for ggplot
  acc_long <- data.frame(
    epoch  = rep(history_df$epoch, 2),
    value  = c(history_df$accuracy, history_df$val_accuracy),
    series = rep(c("Train", "Validation"), each = nrow(history_df))
  )
  loss_long <- data.frame(
    epoch  = rep(history_df$epoch, 2),
    value  = c(history_df$loss, history_df$val_loss),
    series = rep(c("Train", "Validation"), each = nrow(history_df))
  )

  p_acc <- ggplot(acc_long, aes(x = epoch, y = value, colour = series)) +
    geom_line(size = 1.1) +
    geom_point(size = 1.5) +
    scale_y_continuous(labels = percent_format(), limits = c(0, 1)) +
    scale_colour_manual(values = c("Train" = "#2196F3", "Validation" = "#FF5722")) +
    labs(title = "Renart — Accuracy", x = "Epoch", y = "Accuracy", colour = NULL) +
    theme_minimal(base_size = 13) +
    theme(legend.position = "bottom")

  p_loss <- ggplot(loss_long, aes(x = epoch, y = value, colour = series)) +
    geom_line(size = 1.1) +
    geom_point(size = 1.5) +
    scale_colour_manual(values = c("Train" = "#2196F3", "Validation" = "#FF5722")) +
    labs(title = "Renart — Loss", x = "Epoch", y = "Loss", colour = NULL) +
    theme_minimal(base_size = 13) +
    theme(legend.position = "bottom")

  # Combine side by side using base graphics
  acc_path  <- file.path(config$results_dir, "accuracy_curve.png")
  loss_path <- file.path(config$results_dir, "loss_curve.png")
  ggsave(acc_path,  p_acc,  width = 8, height = 5, dpi = 150)
  ggsave(loss_path, p_loss, width = 8, height = 5, dpi = 150)

  cat(sprintf("\n  Accuracy curve → %s\n", acc_path))
  cat(sprintf("  Loss curve     → %s\n",  loss_path))

  # ── Confusion matrix heatmap ──
  cm <- table(
    Predicted = class_names[pred_classes + 1],
    Actual    = class_names[true_labels  + 1]
  )
  cm_df <- as.data.frame(cm)

  p_cm <- ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Freq)) +
    geom_tile(colour = "white") +
    geom_text(aes(label = Freq), size = 2.8, colour = "white") +
    scale_fill_gradient(low = "#1a237e", high = "#f44336") +
    labs(title = "Renart — Confusion Matrix", x = "Actual Artist", y = "Predicted Artist") +
    theme_minimal(base_size = 10) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
      axis.text.y = element_text(size = 8)
    )

  cm_path <- file.path(config$results_dir, "confusion_matrix.png")
  ggsave(cm_path, p_cm, width = 12, height = 10, dpi = 150)
  cat(sprintf("  Confusion matrix → %s\n", cm_path))

  # ── Save class index mapping for predict.js ──
  class_indices <- generators$train$class_indices
  writeLines(
    toJSON(class_indices, auto_unbox = TRUE, pretty = TRUE),
    config$class_index_file
  )
  cat(sprintf("  Class indices    → %s\n\n", config$class_index_file))

  # ── Save final model (non-checkpoint version) ──
  model %>% save_model_hdf5(config$final_model_path)
  cat(sprintf("  Final model      → %s\n", config$final_model_path))
  cat(sprintf("  Best checkpoint  → %s\n\n", config$model_save_path))

  invisible(list(
    val_accuracy = results["accuracy"],
    per_class    = acc_df
  ))
}

# ───────────────────────────────────────────────────────────────────────────────
# 7. Prediction
#    Load a saved model and predict the artist from a single image file.
#    Returns top-3 artists with confidence percentages.
#    Usage: Rscript main.r predict path/to/painting.jpg
# ───────────────────────────────────────────────────────────────────────────────
predict_artwork <- function(image_path, config, top_k = 3) {

  # Validate inputs
  if (!file.exists(image_path)) {
    stop(paste("Image not found:", image_path))
  }
  if (!file.exists(config$model_save_path)) {
    stop(paste("No trained model found at:", config$model_save_path,
               "\nRun training first: Rscript main.r"))
  }
  if (!file.exists(config$class_index_file)) {
    stop(paste("Class index file not found at:", config$class_index_file))
  }

  cat("Loading model...\n")
  model <- load_model_hdf5(config$model_save_path)

  class_indices <- fromJSON(config$class_index_file)
  # Invert: index -> artist name
  idx_to_artist <- setNames(names(class_indices), as.character(unlist(class_indices)))

  # Preprocess image
  img       <- image_load(image_path, target_size = config$image_size)
  img_array <- image_to_array(img)
  img_array <- array_reshape(img_array, c(1, dim(img_array))) / 255.0

  # Predict
  probs     <- model %>% predict(img_array, verbose = 0)
  top_idx   <- order(probs[1, ], decreasing = TRUE)[1:top_k]

  cat("\n══════════════════════════════════════════\n")
  cat(" Renart — Artwork Analysis\n")
  cat("══════════════════════════════════════════\n")
  cat(sprintf(" Image: %s\n\n", basename(image_path)))
  cat(" Top artist predictions:\n")

  for (rank in seq_along(top_idx)) {
    idx        <- top_idx[rank]
    artist     <- idx_to_artist[as.character(idx - 1)]
    confidence <- probs[1, idx] * 100
    bar        <- paste(rep("█", round(confidence / 5)), collapse = "")
    cat(sprintf("  %d. %-28s %5.1f%%  %s\n", rank, artist, confidence, bar))
  }

  cat("══════════════════════════════════════════\n\n")
}

# ───────────────────────────────────────────────────────────────────────────────
# 8. Main entry point
# ───────────────────────────────────────────────────────────────────────────────
main <- function() {

  cat("\n")
  cat("╔══════════════════════════════════════════╗\n")
  cat("║   Renart — Renaissance Art Classifier    ║\n")
  cat("╚══════════════════════════════════════════╝\n\n")

  # Predict-only mode
  if (PREDICT_ONLY) {
    if (is.null(PREDICT_IMAGE) || !nzchar(PREDICT_IMAGE)) {
      stop("Usage: Rscript main.r --predict path/to/painting.jpg")
    }
    predict_artwork(PREDICT_IMAGE, CONFIG)
    return(invisible(NULL))
  }

  # Full training pipeline 

  # Step 1: Dataset
  download_dataset(CONFIG)

  # Step 2: Data generators
  generators <- create_generators(CONFIG)

  # Step 3: Model
  model <- build_model(generators$num_classes, CONFIG)

  # Step 4: Train
  history <- train_model(model, generators, CONFIG)

  # Step 5: Evaluate + save all results
  eval_results <- evaluate_model(model, generators, history, CONFIG)

  cat("╔══════════════════════════════════════════╗\n")
  cat(sprintf("║  Training complete!                      ║\n"))
  cat(sprintf("║  Final val accuracy: %5.2f%%              ║\n",
              eval_results$val_accuracy * 100))
  cat("╚══════════════════════════════════════════╝\n\n")
  cat("Outputs saved to ./results/:\n")
  cat("  accuracy_curve.png    — training progress\n")
  cat("  loss_curve.png        — loss over epochs\n")
  cat("  confusion_matrix.png  — per-artist prediction breakdown\n")
  cat("  class_indices.json    — artist label map (used by predict.js)\n")
  cat("  training_log.csv      — raw epoch-by-epoch metrics\n\n")
  cat("To predict a new image:\n")
  cat("  Rscript main.r --predict path/to/painting.jpg\n\n")
}

# Run
main()
