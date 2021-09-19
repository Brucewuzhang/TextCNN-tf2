import tensorflow as tf


class ConvMaxPooling1d(tf.keras.layers.Layer):
    def __init__(self, filters, kernel):
        super(ConvMaxPooling1d, self).__init__()
        self.kernel_size = kernel
        self.conv = tf.keras.layers.Conv1D(filters, kernel, activation='relu')
        self.pool = tf.keras.layers.GlobalMaxPool1D()

    def call(self, inputs, masks=None):
        conv_out = self.conv(inputs)
        if masks is not None:
            masks_exp = tf.expand_dims(masks, axis=-1)
            conv_out += masks_exp[:, self.kernel_size - 1:]
        pool_out = self.pool(conv_out)
        return pool_out


class TextCNN(tf.keras.models.Model):
    def __init__(self, embedding_size, hidden_size, num_classes, filters_list=[30, 40, 50], kernels=[3, 4, 5],
                 dropout=0.5):
        super(TextCNN, self).__init__()
        self.embedding = tf.keras.layers.Embedding(1000, embedding_size)
        self.conv_maxs = [ConvMaxPooling1d(f, k) for f, k in zip(filters_list, kernels)]
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.dense = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.classifier = tf.keras.layers.Dense(num_classes)

    def call(self, inputs, training=None, mask=None):
        input_ids = inputs['input_ids']
        mask = inputs.get('attention_mask', None)
        if mask is not None:
            mask = tf.cast(tf.equal(mask, 0), dtype=tf.float32) * float('-inf')
        embeddings = self.embedding(input_ids)

        conv_outs = [layer(embeddings, mask) for layer in self.conv_maxs]
        concat_out = tf.concat(conv_outs, axis=-1)
        dense_out = self.dense(concat_out)
        drop_out = self.dropout(dense_out, training=training)
        logits = self.classifier(drop_out)

        return logits


if __name__ == '__main__':
    import numpy as np

    model = TextCNN(64, 64, 2)
    model.compile()

    e1 = np.random.randint(0, 1000, size=(32, 30), dtype=np.int32)
    e2 = np.random.randint(0, 1000, size=(16, 50), dtype=np.int32)
    e3 = np.random.randint(0, 1000, size=(32, 48), dtype=np.int32)
    masks = np.random.randint(0, 2, size=(32, 48), dtype=np.int32)
    model({"input_ids": e1}, training=False)
    model({"input_ids": e2}, training=False)
    out = model({"input_ids": e3, "attention_mask": masks}, training=False)
    print(out)
