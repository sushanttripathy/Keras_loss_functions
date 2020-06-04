import tensorflow as tf
from keras import backend as K

def class_weighted_focal_loss(class_weights, gamma=.5):
    class_weights = tf.constant(class_weights, tf.float32)
    gamma = float(gamma)

    def focal_loss_function(y_true, y_pred):
        """
        Focal loss for multi-label classification.
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} : Ground truth labels, with shape (batch_size, number_of_classes).
            y_pred {tensor} : Model's predictions, with shape (batch_size, number_of_classes).

        Keyword Arguments:
            gamma {float} : The Gamma parameter in Focal Loss. Default value (0.5).
            class_weights {list[float]} : Non-zero, positive class-weights. This is used instead 
                                          of Alpha parameter.

        Returns:
            loss {tensor} : Single dimensional tensor. Hence, effectively a scalar.
        """
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)
        
        predictions_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        predictions_0 = tf.clip_by_value(prediction_0, K.epsilon(), 1.0-K.epsilon())
        
        predictions_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        predictions_1 = tf.clip_by_value(predictions_1, K.epsilon(), 1.0-K.epsilon())

        
        cross_entropy_1 = tf.multiply(y_true, -tf.log(predictions_1))
        cross_entropy_0 = tf.multiply(tf.subtract(1.0, y_true), -tf.log(tf.subtract(1.0, predictions_0)))
        cross_entropy = tf.add(cross_entropy_1, cross_entropy_0)
        class_weighted_cross_entropy = tf.multiply(cross_entropy, class_weights)
        
        weight_1 = tf.multiply(y_true, tf.pow(tf.subtract(1., predictions_1), gamma))
        weight_0 = tf.multiply(tf.subtract(1., y_true), tf.pow(predictions_0, gamma))
        
        weight = tf.add(weight_0, weight_1)
        
        focal_loss_tensor = tf.multiply(weight, class_weighted_cross_entropy)
        focal_loss = tf.reduce_mean(focal_loss_tensor)
        return focal_loss
    
    return focal_loss_function
