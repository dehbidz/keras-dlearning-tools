from tensorflow.keras import backend as K

def categorical_focal_loss(y, y_pred):
    gamma = 2.5  # 2.5 and 0.5 are values set on some paper
    alpha = 0.5  # 2.0 and 0.25
    # paper multiclass classification
    def focal_loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * y_true * K.pow((1-y_pred), gamma)
        loss = weight * cross_entropy
        loss = K.sum(loss, axis=1)
        return loss
    return focal_loss(y, y_pred)
