import tensorflow as tf

def configure_denoiser_run(model_dir, train_record_path, eval_record_path, data_input_fn, cnn_model_fn):
    run_config = tf.estimator.RunConfig(
        model_dir=model_dir,
        save_checkpoints_steps=20,
        save_summary_steps=20,
        tf_random_seed=0
    )

    hparams = {
        'learning_rate': 1e-3,
        'dropout_rate': 0.4,
        'data_directory': model_dir
    }

    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,            # Model to be trained
        config=run_config,
        params=hparams
    )

    train_batch_size = 32

    train_input_fn = data_input_fn(tfrecords_path=train_record_path,
                                   batch_size=train_batch_size,
                                   shuffle=True)
    eval_input_fn = data_input_fn(tfrecords_path=eval_record_path,
                                  batch_size=10,                    # Num eval samples
                                  shuffle=True)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=40)

    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                      steps=100,
                                      start_delay_secs=0)

    return classifier, train_spec, eval_spec


def speech_enhancement_cnn():
    return