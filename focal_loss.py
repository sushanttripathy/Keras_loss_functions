import tensorflow as tf
from keras import backend as K


def class_weighted_focal_loss(class_weights, gamma=2.0, class_sparsity_coefficient=10.0):
    class_weights = K.constant(class_weights, tf.float32)
    gamma = K.constant(gamma, tf.float32)
    class_sparsity_coefficient = K.constant(class_sparsity_coefficient, tf.float32)

    def focal_loss_function(y_true, y_pred):
        """
        Focal loss for multi-label classification.
        https://arxiv.org/abs/1708.02002
        Arguments:
            y_true {tensor} : Ground truth labels, with shape (batch_size, number_of_classes).
            y_pred {tensor} : Model's predictions, with shape (batch_size, number_of_classes).
        Keyword Arguments:
            class_weights {list[float]} : Non-zero, positive class-weights. This is used instead
                                          of Alpha parameter.
            gamma {float} : The Gamma parameter in Focal Loss. Default value (2.0).
            class_sparsity_coefficient {float} : The weight of True labels over False labels. Useful
                                                 if True labels are sparse. Default value (1.0).
        Returns:
            loss {tensor} : A tensor of focal loss.
        """

        predictions_0 = (1.0 - y_true) * y_pred
        predictions_1 = y_true * y_pred

        cross_entropy_0 = (1.0 - y_true) * (-K.log(K.clip(1.0 - predictions_0, K.epsilon(), 1.0 - K.epsilon())))
        cross_entropy_1 = y_true *(class_sparsity_coefficient * -K.log(K.clip(predictions_1, K.epsilon(), 1.0 - K.epsilon())))

        cross_entropy = cross_entropy_1 + cross_entropy_0
        class_weighted_cross_entropy = cross_entropy * class_weights

        weight_1 = K.pow(K.clip(1.0 - predictions_1, K.epsilon(), 1.0 - K.epsilon()), gamma)
        weight_0 = K.pow(K.clip(predictions_0, K.epsilon(), 1.0 - K.epsilon()), gamma)

        weight = weight_0 + weight_1

        focal_loss_tensor = weight * class_weighted_cross_entropy

        return K.mean(focal_loss_tensor, axis=1)

    return focal_loss_function
